from comformer import train_from_extxyz

results = train_from_extxyz(
    extxyz_file="./mp_dielectric.extxyz",
    target_property="total_dielectric_constant",
    cache_graphs=True,
    graph_cache_dir="./graph_cache",
    output_dir="./e_total",
    n_epochs=20,
    batch_size=64,
)
