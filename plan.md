Goal: I have a lerobot dataset. I want a script pi05_extract_vision_embeddings.bash that I can run, mirroring the settings of pi05_training.bash. The script modifies the lerobot dataset to include vision embeddings of all camera images with each frame.

The purpose of this is to add vision embeddings to the dataset for a separate application.

I want to load a lerobot dataset and sample each frame, in order, no random sampling necessary. Then pass all camera images through the vision encoder and save those embeddings to the dataset.

Of course, one pass through the dataset is all that is necessary.

Minimal diff please.