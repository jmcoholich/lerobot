from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

dataset1 = "walle_partial_success"
dataset2 = "skywalker_partial_success"

merged_dataset_name = "walle_skywalker_partial_success"


aggregate_datasets(
    repo_ids=[
        f"lerobot/{dataset1}",
        f"lerobot/{dataset2}",
    ],
    roots=[
        Path(f"/data3/lerobot_data/{dataset1}"),
        Path(f"/data3/lerobot_data/{dataset2}"),
    ],
    aggr_repo_id=f"lerobot/{merged_dataset_name}",
    aggr_root=Path(f"/data3/lerobot_data/{merged_dataset_name}"),
    video_files_size_in_mb=1,
)