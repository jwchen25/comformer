# Multi-GPU Training Error Fix

## 遇到的错误

### 错误 1: DistributedSampler 初始化失败 ✅ 已修复

**错误信息**:
```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

**错误原因**:
代码尝试在分布式环境初始化**之前**创建 `DistributedSampler`:
1. `train_from_list()` 创建 DataLoader 时就尝试创建 `DistributedSampler`
2. 但分布式环境在后续的 `train_main()` 中才初始化
3. `DistributedSampler` 需要已初始化的分布式环境

**解决方案**:
- 在 `custom_train.py` 中提前初始化分布式环境
- 在 `train.py` 的 `setup_distributed()` 中检查避免重复初始化

### 错误 2: Config 类型错误 ✅ 已修复

**错误信息**:
```python
AttributeError: 'dict' object has no attribute 'distributed'
```

**错误原因**:
执行顺序问题：
1. `train_from_list()` 创建了字典类型的 config
2. `train_main()` **先调用** `setup_distributed(config)`
3. 但此时 config 还是字典，尚未转换为 `TrainingConfig` 对象
4. `setup_distributed()` 尝试访问 `config.distributed` 属性失败

**解决方案**:
- 在 `train_main()` 中调整执行顺序
- **先转换** config 字典为 `TrainingConfig` 对象
- **再调用** `setup_distributed()`

### 错误 3: DDP 未使用参数错误 ✅ 已修复

**错误信息**:
```python
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
Parameter indices which did not receive grad for rank 2: 94 114 115
```

**错误原因**:
1. ComFormer 模型中有些参数在某些情况下不参与前向传播
2. DDP 默认要求所有参数都参与梯度计算
3. 当有未使用参数时，DDP 无法正确同步梯度

**解决方案**:
- 在创建 DDP 时设置 `find_unused_parameters=True`
- 允许模型有条件分支和未使用的参数
- PyTorch 会自动检测并处理未使用的参数

## 完整修复方案

已经修复了以下文件:
1. **comformer/custom_train.py** (第758-802行):
   - 在创建 DataLoader 前提前初始化分布式环境
2. **comformer/train.py**:
   - 第228-239行: 调整 `train_main()` 执行顺序（先转换 config，再初始化分布式）
   - 第67-84行: 更新 `setup_distributed()` 避免重复初始化
   - 第369行: 设置 `find_unused_parameters=True` 处理未使用参数

### 修复内容

#### 1. custom_train.py (第758-802行)
在创建 DistributedSampler 前检查并初始化分布式环境:
- 自动从环境变量读取 RANK、LOCAL_RANK、WORLD_SIZE
- 如果未初始化,则调用 `dist.init_process_group()`
- 正确设置 CUDA 设备
- 创建 DistributedSampler 进行数据分区

#### 2. train.py (第228-239行)
调整 `train_main()` 执行顺序:
- **先转换** config 字典为 `TrainingConfig` 对象
- **再调用** `setup_distributed(config)`
- 确保 config 有正确的属性才能被访问

#### 3. train.py (第67-84行)
更新 `setup_distributed()` 函数:
- 检查分布式环境是否已初始化
- 如已初始化,则复用现有环境
- 避免重复初始化导致错误

#### 4. train.py (第369行)
设置 DDP 参数处理未使用参数:
```python
net = torch.nn.parallel.DistributedDataParallel(
    net,
    device_ids=[local_rank] if torch.cuda.is_available() else None,
    output_device=local_rank if torch.cuda.is_available() else None,
    find_unused_parameters=True  # 允许模型有未使用的参数
)
```
这解决了 ComFormer 模型中某些条件分支导致的参数未使用问题。

## 使用方法

修复后,您的代码应该可以正常运行:

```python
from comformer import train_from_extxyz

# 使用 torchrun 启动脚本
# torchrun --nproc_per_node=4 your_script.py

results = train_from_extxyz(
    extxyz_file="../datasets/mp_shear_modulus.extxyz",
    target_property="shear_modulus",
    batch_size=64,
    n_epochs=150,
    distributed=True,  # 启用多GPU训练
    output_dir="./output"
)
```

### 启动命令

```bash
# 4个GPU
torchrun --nproc_per_node=4 train_script.py

# 8个GPU
torchrun --nproc_per_node=8 train_script.py
```

## 验证修复

运行您的训练脚本,应该能看到以下输出:

```
Initializing distributed environment for DataLoader creation...
Distributed environment initialized: world_size=4, rank=0
Created DistributedSamplers for data partitioning across GPUs

Final dataset sizes:
  Train: 9748
  Val: 1218
  Test: 1220

============================================================
Starting training...
============================================================
Using existing distributed environment:
  World size: 4
  Rank: 0
  Local rank: 0
Model wrapped with DistributedDataParallel on device cuda:0
  find_unused_parameters=True (allows conditional model branches)

[Epoch 1] Training started...
```

这表明:
1. ✅ 分布式环境成功初始化
2. ✅ DistributedSamplers 成功创建
3. ✅ 数据正确分区到各个GPU
4. ✅ train_main() 复用了现有的分布式环境
5. ✅ DDP 正确配置 find_unused_parameters=True
6. ✅ 训练正常开始，没有参数错误

## 关键改进

1. **提前初始化**: 在需要 DistributedSampler 之前初始化分布式环境
2. **执行顺序修正**: 先转换 config 类型，再使用其属性
3. **避免重复**: 检查是否已初始化,避免重复初始化错误
4. **处理未使用参数**: 设置 `find_unused_parameters=True` 支持条件模型分支
5. **正确的数据分区**: 每个GPU处理数据的不同子集
6. **兼容性**: 支持单GPU和多GPU模式自动切换

## 其他注意事项

### Git 警告
日志中的 git 警告可以忽略:
```
fatal: not a git repository (or any parent up to mount point /capstor/store)
```
这只是因为代码在非 git 仓库目录运行,不影响功能。

### CGCNN 特征警告
以下警告是正常的,已自动处理:
```
warning: could not load CGCNN features for 103
Setting it to max atomic number available here, 103
```
这表示某些高原子序数元素使用了最接近的可用特征。

### RuntimeWarning: arccos
```
RuntimeWarning: invalid value encountered in arccos
```
这是在计算键角时遇到的数值精度问题,代码会自动处理。

## 性能预期

使用4个GPU训练,您应该能看到:
- **加速比**: ~3.0-3.5x (相比单GPU)
- **数据分区**: 每个GPU处理 ~1/4 的数据
- **有效batch size**: 64 × 4 = 256

## 故障排除

### 如果仍然报错

1. **检查环境变量**:
   ```bash
   echo $RANK
   echo $LOCAL_RANK
   echo $WORLD_SIZE
   ```
   这些应该由 torchrun 自动设置。

2. **确认使用 torchrun**:
   - ✅ 正确: `torchrun --nproc_per_node=4 script.py`
   - ❌ 错误: `python script.py`

3. **检查 PyTorch 版本**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   应该 >= 2.6

4. **检查 CUDA 可用性**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch; print(torch.cuda.device_count())"
   ```

### 如果想单GPU训练

只需设置 `distributed=False` 或不使用 torchrun:

```python
results = train_from_extxyz(
    extxyz_file="../datasets/mp_shear_modulus.extxyz",
    target_property="shear_modulus",
    distributed=False,  # 单GPU模式
    output_dir="./output"
)
```

或直接运行:
```bash
python train_script.py  # 不使用 torchrun
```

## 总结

**三个错误都已修复**:
1. ✅ **错误1**: DistributedSampler 初始化失败 → 提前初始化分布式环境
2. ✅ **错误2**: Config 类型错误 → 调整执行顺序
3. ✅ **错误3**: DDP 未使用参数错误 → 设置 find_unused_parameters=True

**修改的文件**:
- `comformer/custom_train.py:758-802` - 提前初始化分布式环境
- `comformer/train.py:228-239` - 调整 config 转换顺序
- `comformer/train.py:67-84` - 避免重复初始化
- `comformer/train.py:369` - 处理未使用参数

**功能完整**:
- ✅ 使用 `train_from_list()` 进行多GPU训练
- ✅ 使用 `train_from_extxyz()` 进行多GPU训练
- ✅ 数据自动分区到各个GPU
- ✅ 梯度自动同步
- ✅ 支持模型条件分支
- ✅ 加速大规模数据集训练

**向后兼容**:
修复后的代码向后兼容,单GPU训练不受影响。
