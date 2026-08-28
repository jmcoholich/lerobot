from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

datasets = [
    "walle_skywalker_testset",
    "plug5_offline_rl_dataset",
]

merged_dataset_name = "plug5_offline_rl_dataset_walle_skywalker_testset"

DATASET_DIR = Path("/coc/testnvme/jcoholich3/lerobot_data")
# DATASET_DIR = Path("/data3/lerobot_data")

aggregate_datasets(
    repo_ids=[f"lerobot/{name}" for name in datasets],
    roots=[DATASET_DIR / name for name in datasets],
    aggr_repo_id=f"lerobot/{merged_dataset_name}",
    aggr_root=DATASET_DIR / merged_dataset_name,
    video_files_size_in_mb=1,
)