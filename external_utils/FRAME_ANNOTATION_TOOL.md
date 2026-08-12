# Frame annotation tool

This tool adds a string feature to every frame in a LeRobot dataset and provides a browser UI for editing it.
Existing annotation values are preserved.

## Start the tool

Run this on the machine that stores the dataset:

```bash
python external_utils/annotation_tool.py /path/to/dataset \
  --host 127.0.0.1 \
  --port 7860
```

The server initializes a missing `annotation` column automatically. To initialize it separately:

```bash
python external_utils/add_annotations_to_lerobot_dataset.py /path/to/dataset
```

The default quick labels are `plug picked`, `success`, `failure`, and `partial success`. Override them with:

```bash
python external_utils/annotation_tool.py /path/to/dataset \
  --labels "plug picked,plug inserted,success,failure,partial success"
```

Labels used in the browser are remembered and shown first. The numbered label buttons can also be applied with
keys `1` through `9`. Left and right arrow keys move between frames when the annotation text field is not focused.
The label order is stable: new labels are appended, using a label does not move it, and labels can be dragged to save
a new order.

Each edit is immediately persisted to `meta/annotation_drafts.sqlite3`. The **Write dataset** action atomically
rewrites affected Parquet files. Episode navigation preserves drafts, and **Discard pending** reverts all drafts to
the last committed Parquet values. This keeps individual frame edits responsive without risking unsaved browser
state. Keep the SQLite file with the dataset while annotation is in progress; it is not needed after all pending
changes have been written.

Use **-10**, **-1**, **+1**, and **+10** to move around without annotating. The episode scrubber has clickable green
marks for committed annotations and amber marks for pending annotations.

The default **Front + Side** view shows synchronized, centered square crops from both cameras. Individual **Front**,
**Side**, and **Wrist** views are also available. Applying an annotation keeps the viewer on the current frame.

## Remote server

Keep the annotation server bound to `127.0.0.1` on the remote machine, then forward its port from your workstation:

```bash
ssh -L 7860:127.0.0.1:7860 user@remote-server
```

Open `http://127.0.0.1:7860` locally. Video frames are decoded on the remote server and sent as JPEG images, so the
dataset and its video files do not need to be copied to the workstation.

## Data safety

Adding the column and committing annotations rewrites Parquet files because Parquet does not support in-place column
updates. Writes use a temporary file in the same directory followed by an atomic replacement. Make a dataset backup
before the first initialization, especially when annotating the only copy of collected data.
