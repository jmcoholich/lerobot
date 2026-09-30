The launcher automatically activates the `lerobot` conda environment:

```bash
bash pi05_value_inference_awr.bash --dry-run
bash pi05_value_inference_awr.bash
```

Edit the variables at the top of the launcher, or override them through the environment:

```bash
RUN_TEMPLATE='off_value_fn_random_gamma099_unfrozen_dropout0_fold_{fold}' \
CHECKPOINT=best MAX_PARALLEL=10 bash pi05_value_inference_awr.bash
```

The defaults are `off_value_fn_random_unfrozen_dropout0_fold_{fold}`, checkpoint `last`,
and `plug5_offline_rl_dataset_walle_skywalker_testset_annotated`. `MAX_PARALLEL` is empty
by default (no array concurrency cap). `DATASET_ROOT`, `OUTPUT_DIR`, and `NAME` are also
configurable. Only `--dry-run` is a launcher flag; settings use environment variables.

The launcher submits five independent arrays through `pi05_value_inference_static.bash`,
then `pi05_value_merge_awr.bash` after every array succeeds. Resource requests are in those
scripts' `#SBATCH` headers. Python validates the saved checkpoint splits and return targets,
and later merges predictions and generates statistics; Bash handles all job submission.

Each episode uses the model whose saved `test_episodes` contains it. The default columns are:

- `awr_value_off_value_fn_random_unfrozen_dropout0_last_OOF`
- `awr_advantage_off_value_fn_random_unfrozen_dropout0_last_OOF`

Advantages are `dataset[checkpoint.value_key] - raw_value_prediction`, with no clipping or
standardization. The default return key is `annotation_return_gamma_0.95`. Inference reverses
checkpoint value normalization when applicable. Validation rejects overlapping/incomplete
folds, different training datasets, inconsistent returns, and bootstrapped training targets.

The printed report directory under `outputs/awr/` contains the plan, fold inputs, job IDs,
per-episode predictions, `stats.json`, and `histograms.png`. Statistics cover dataset size,
episode lengths, returns, values, advantages, prediction errors, and per-fold summaries.
`episode_plots/episode_000000.png` through `episode_000004.png` show the first five dataset
episodes: ground-truth returns, OOF values, and advantages versus frame index. Each plot
includes the dataset, original `fname`, episode index, model/checkpoint, return column,
frame count, duration, and prediction error. These are generated automatically by the merge job.

Wait for the merge to succeed (`complete.json`) before training. It updates parquet columns
and dataset feature/statistics metadata, retaining originals under `meta/awr_backups/`.
Use a new `NAME` for subsequent annotations; existing columns are never overwritten.
To retry merging after fixing failed inference: `sbatch pi05_value_merge_awr.bash /path/to/plan.json`.

Pass the advantage column as the fourth argument to AWR training:

```bash
cd /coc/testnvme/jcoholich3/lerobot
sbatch pi05_training.bash \
  plug5_offline_rl_dataset_walle_skywalker_testset_annotated \
  pi05_awr_oof 1.0 \
  awr_advantage_off_value_fn_random_unfrozen_dropout0_last_OOF
```
