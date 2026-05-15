from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


aggregate_datasets(
    repo_ids=[
        "lerobot/thread3_bc_and_dagger",
        "lerobot/unthread4_bc_and_dagger",
    ],
    roots=[
        Path("/data3/lerobot_data/thread3_bc_and_dagger"),
        Path("/data3/lerobot_data/unthread4_bc_and_dagger"),
    ],
    aggr_repo_id="lerobot/thread3_unthread4_bc_and_dagger_FIXED",
    aggr_root=Path("/data3/lerobot_data/thread3_unthread4_bc_and_dagger_FIXED"),
    video_files_size_in_mb=1,
)
