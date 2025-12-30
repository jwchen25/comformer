import os
from functools import partial
from typing import Any, Dict, Union
import pickle as pk
import json
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from jarvis.db.jsonutils import dumpjson

import torch
from torch import nn

import ignite
from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError

from comformer.config import TrainingConfig
from comformer.models.comformer import iComformer, eComformer


# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def setup_distributed(config: TrainingConfig):
    """
    Initialize distributed training environment.

    For PyTorch >= 2.6, uses environment variables set by torchrun:
    - RANK: Global rank
    - LOCAL_RANK: Local rank (GPU index on current node)
    - WORLD_SIZE: Total number of processes

    Args:
        config: Training configuration

    Returns:
        Tuple of (rank, local_rank, world_size, is_distributed)
    """
    if not config.distributed:
        return 0, 0, 1, False

    # Check if already initialized (might be initialized early in train_from_list)
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", config.local_rank))

        # Update config with actual values
        config.rank = rank
        config.local_rank = local_rank
        config.world_size = world_size

        if rank == 0:
            print(f"Using existing distributed environment:")
            print(f"  World size: {world_size}")
            print(f"  Rank: {rank}")
            print(f"  Local rank: {local_rank}")

        return rank, local_rank, world_size, True

    # Get distributed parameters from environment variables (set by torchrun)
    rank = int(os.environ.get("RANK", config.rank))
    local_rank = int(os.environ.get("LOCAL_RANK", config.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", config.world_size))

    # Update config with actual values
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    # Initialize process group
    torch.distributed.init_process_group(
        backend=config.dist_backend,
        init_method=config.dist_url,
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Distributed training initialized:")
        print(f"  Backend: {config.dist_backend}")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")

    return rank, local_rank, world_size, True


def cleanup_distributed():
    """Clean up distributed training process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process (rank 0)."""
    return rank == 0


class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def count_parameters(model):
        total_params = 0
        for parameter in model.parameters():
            total_params += parameter.element_size() * parameter.nelement()
        for parameter in model.buffers():
            total_params += parameter.element_size() * parameter.nelement()
        total_params = total_params / 1024 / 1024
        print(f"Total size: {total_params}")
        print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return total_params

def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_main(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    # Setup distributed training first
    rank, local_rank, world_size, is_distributed = setup_distributed(config)
    is_main = is_main_process(rank)

    # Set device based on distributed setup
    if is_distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Only print from main process
    if is_main:
        print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            if is_main:
                print("Check", exp)
                print('error in converting to training config!')
            # If validation fails, raise the error to prevent further issues
            raise ValueError(
                f"Configuration validation failed: {exp}\n"
                "Please check your configuration parameters."
            ) from exp
    import os

    # Only main process creates output directory
    if is_main and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Synchronize all processes
    if is_distributed:
        torch.distributed.barrier()
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    # Pydantic v2 compatibility: .dict() -> .model_dump()
    # try:
    #     tmp = config.model_dump()
    # except AttributeError:
    #     tmp = config.dict()
    # f = open(os.path.join(config.output_dir, "config.json"), "w")
    # f.write(json.dumps(tmp, indent=4))
    # f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir

    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = True
    mean_train = None
    std_train = None

    # Only custom datasets with pre-built loaders are supported
    if not train_val_test_loaders:
        raise ValueError(
            "train_val_test_loaders must be provided. "
            "For custom datasets, use train_from_list() from comformer.custom_train module. "
            "Predefined datasets (dft_3d, megnet, mpf) are no longer supported."
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
        # Optional mean and std for custom loaders
        if len(train_val_test_loaders) > 4:
            mean_train = train_val_test_loaders[4]
        if len(train_val_test_loaders) > 5:
            std_train = train_val_test_loaders[5]

    if mean_train is None:
        mean_train = 0.0
        if is_main:
            print('mean train is none! set to 0.0!')
    else:
        if is_main:
            print('mean train:', mean_train)
    if std_train is None:
        std_train = 1.0
        if is_main:
            print('std train is none! set to 1.0!')
    else:
        if is_main:
            print('std train:', std_train)

    # Save normalization statistics to config if available
    config.mean_train = mean_train
    config.std_train = std_train

    # Save config with normalization statistics (only on main process)
    if is_main:
        try:
            tmp = config.model_dump()
        except AttributeError:
            tmp = config.dict()
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            f.write(json.dumps(tmp, indent=4))
        print("config:")
        pprint.pprint(tmp)
        print(f"Saved config.json with normalization statistics: mean_train={mean_train}, std_train={std_train}")

    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "iComformer" : iComformer,
        "eComformer" : eComformer,
    }
    if model is None:
        net = _model.get(config.model.name)(config.model)
        if is_main:
            print("config:")
            # Pydantic v2 compatibility: .dict() -> .model_dump()
            try:
                pprint.pprint(config.model.model_dump())
            except AttributeError:
                pprint.pprint(config.model.dict())
    else:
        net = model

    net.to(device)

    # Wrap model with DistributedDataParallel if distributed training is enabled
    if is_distributed:
        # For PyTorch >= 2.6, use DDP with proper device_ids
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )
        if is_main:
            print(f"Model wrapped with DistributedDataParallel on device {device}")

    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )
    elif config.scheduler == "polynomial":
        steps_per_epoch = len(train_loader)
        total_iter = steps_per_epoch * config.epochs
        scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=0.0005, end_lr=0.00001, power=1)

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    criterion = criteria[config.criterion]
    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )
    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # Only print model parameters from main process
    if is_main:
        count_parameters(net)

    # Checkpoint saving (only on main process)
    if config.write_checkpoint and is_main:
        # model checkpointing - save every epoch
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            # filename_prefix="last",
            # n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

        # Save best model as best_model.pt
        # Track best score
        best_score = {"value": float('-inf')}

        def save_best_model(engine):
            """Save best model with fixed filename 'best_model.pt'"""
            score = engine.state.metrics.get("neg_mae", float('-inf'))
            if score > best_score["value"]:
                best_score["value"] = score
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                # Save model state dict
                checkpoint_dict = {"model": net.state_dict()}
                torch.save(checkpoint_dict, best_model_path)
                if config.progress:
                    print(f"\n✓ Saved best model to {best_model_path} (MAE: {-score:.4f})")

        evaluator.add_event_handler(Events.EPOCH_COMPLETED, save_best_model)

    # Progress bar (only on main process)
    if config.progress and is_main:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        evaluator.run(val_loader)

        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)



        epoch_num = len(history["validation"][t_metric])
        if epoch_num % 20 == 0:
            train_evaluator.run(train_loader)
            tmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = tmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
        else:
            tmetrics = {}
            tmetrics['mae'] = -1

        # Only save outputs from main process
        if config.store_outputs and is_main:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )

        # Only print progress from main process
        if config.progress and is_main:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        if classification:
            my_metrics = "accuracy"
        else:
            my_metrics = "neg_mae"

        def default_score_fn(engine):
            score = engine.state.metrics[my_metrics]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    # optionally log results to tensorboard (only on main process)
    if config.log_tensorboard and is_main:
        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator_tb in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator_tb,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # Run training
    trainer.run(train_loader, max_epochs=config.epochs)

    # TensorBoard cleanup (only on main process)
    if config.log_tensorboard and is_main:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()

    # Prediction writing (only on main process)
    if config.write_predictions and classification and is_main:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
        and is_main
    ):
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
        dumpjson(
            filename=os.path.join(
                config.output_dir, "multi_out_predictions.json"
            ),
            data=mem,
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
        and is_main
    ):
        net.eval()
        targets = []
        predictions = []
        import time
        t1 = time.time()
        with torch.no_grad():
            from tqdm import tqdm
            for dat in tqdm(test_loader):
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                out_data = out_data.cpu().numpy().flatten().tolist()
                target = target.cpu().numpy().flatten().tolist()
                # if len(target) == 1:
                #     target = target[0]
                targets.extend(target)
                predictions.extend(out_data)
        t2 = time.time()
        from sklearn.metrics import mean_absolute_error
        targets = np.array(targets) * std_train + mean_train
        predictions = np.array(predictions) * std_train + mean_train
        test_mae = mean_absolute_error(targets, predictions)
        print("Test MAE:", test_mae)

        # Save predictions to CSV file
        # Get sample IDs from test dataset
        if hasattr(test_loader.dataset, 'ids'):
            sample_ids = test_loader.dataset.ids.tolist()
        else:
            sample_ids = list(range(len(targets)))

        # Flatten predictions if needed
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Create DataFrame with predictions and targets
        results_df = pd.DataFrame({
            'id': sample_ids,
            'target': targets_flat,
            'prediction': predictions_flat
        })

        # Save to CSV file in output_dir
        csv_path = os.path.join(config.output_dir, 'test_predictions.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

        # Create correlation plot with metrics
        # Calculate R2 score
        r2 = r2_score(targets_flat, predictions_flat)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((targets_flat - predictions_flat) ** 2))

        # Create figure
        plt.figure(figsize=(8, 8))

        # Scatter plot
        plt.scatter(targets_flat, predictions_flat, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

        # Add reference line (perfect prediction)
        min_val = min(targets_flat.min(), predictions_flat.min())
        max_val = max(targets_flat.max(), predictions_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # Add labels and title
        plt.xlabel('True Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.title('Test Set: Predicted vs True Values', fontsize=14, fontweight='bold')

        # Add metrics as text annotation
        metrics_text = f'R² = {r2:.4f}\nMAE = {test_mae:.4f}\nRMSE = {rmse:.4f}\nN = {len(targets_flat)}'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot as JPG
        plot_path = os.path.join(config.output_dir, 'test_predictions_correlation.jpg')
        plt.savefig(plot_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation plot saved to: {plot_path}")

    # Clean up distributed training
    if is_distributed:
        cleanup_distributed()

    return history

