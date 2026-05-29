import json
from pathlib import Path

import lerobot.datasets.aggregate as aggregate


dagger_info_path = Path("/data3/lerobot_data/thread3_dagger_only/meta/info.json")
original_dagger_info = dagger_info_path.read_text()
original_update_data_df = aggregate.update_data_df


def fill_missing_gripper_pos(df):
    state_key = "observation.state"
    if state_key not in df:
        return df

    first_valid_gripper_pos_by_episode = {}
    for _, row in df.iterrows():
        state = row[state_key]
        if state is None:
            continue

        gripper_pos = state[7]
        if not aggregate.pd.isna(gripper_pos):
            first_valid_gripper_pos_by_episode.setdefault(row["episode_index"], gripper_pos)

    for row_idx, row in df.iterrows():
        state = row[state_key]
        if state is None or not aggregate.pd.isna(state[7]):
            continue

        fixed_state = list(state)
        fixed_state[7] = first_valid_gripper_pos_by_episode[row["episode_index"]]
        df.at[row_idx, state_key] = fixed_state

    return df


def update_data_df_without_fname(df, src_meta, dst_meta):
    df = original_update_data_df(df, src_meta, dst_meta)
    df = df.drop(columns=["fname"], errors="ignore")
    return fill_missing_gripper_pos(df)


try:
    dagger_info = json.loads(original_dagger_info)
    dagger_info["features"].pop("fname", None)
    dagger_info_path.write_text(json.dumps(dagger_info, indent=4))
    aggregate.update_data_df = update_data_df_without_fname

    aggregate.aggregate_datasets(
        repo_ids=[
            "lerobot/thread3",
            "lerobot/thread3_dagger_only",
        ],
        roots=[
            Path("/data3/lerobot_data/thread3"),
            Path("/data3/lerobot_data/thread3_dagger_only"),
        ],
        aggr_repo_id="lerobot/thread3_bc_and_only_dagger",
        aggr_root=Path("/data3/lerobot_data/thread3_bc_and_only_dagger"),
        video_files_size_in_mb=1,
    )
finally:
    aggregate.update_data_df = original_update_data_df
    dagger_info_path.write_text(original_dagger_info)
