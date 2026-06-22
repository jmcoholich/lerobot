from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

datasets = [
    "walle_skywalker_success",
    "walle_skywalker_partial_success",
    "walle_skywalker_failure",
]

merged_dataset_name = "walle_skywalker_testset"

aggregate_datasets(
    repo_ids=[f"lerobot/{name}" for name in datasets],
    roots=[Path(f"/data3/lerobot_data/{name}") for name in datasets],
    aggr_repo_id=f"lerobot/{merged_dataset_name}",
    aggr_root=Path(f"/data3/lerobot_data/{merged_dataset_name}"),
    video_files_size_in_mb=1,
)