# AuthenticVision v4 — Training fix in plain language

## Why the first model performed badly on your images

The first run used `HOLDOUT_GENERATOR = 'Midjourney'`. That setting intentionally removed Midjourney fake images from training so the model could be tested against an unseen generator. We then put that research-test model into the app as if it were the deployable model. That was the wrong choice for a general-purpose detector. The first run also used only up to 1,000 AI images per training generator, 3 epochs, and a single `1e-4` learning rate for the whole pretrained model.

Your scores also show real and AI examples overlap. Changing the threshold cannot separate images whose scores overlap, so the old low custom threshold was not a proper fix.

## What v4 changes

1. `HOLDOUT_GENERATOR = 'all'`: train on every generator family available in Tiny-GenImage's training split, including Midjourney.
2. Up to 2,000 AI training images per available generator and 8 maximum epochs.
3. Smaller learning rate for the pretrained image backbone (`1e-5`) and a larger learning rate for the classifier head (`1e-4`).
4. Mild JPEG compression/color augmentation to reduce sensitivity to typical image saving and compression.
5. Label smoothing and a validation-loss learning-rate scheduler.
6. Independent, image-disjoint validation and test subsets. The test is not an unseen-generator test, because these generator families appear in training.
7. Google Drive mount is optional. If authentication fails, outputs go to temporary Colab storage, with a final results ZIP download cell.

## How to run this version

1. Download `AuthenticVision_GenImage_Trainer_v4_corrected.zip` and `AuthenticVision_Cloud_GPU_Training_v4.ipynb`.
2. Open the v4 notebook in Colab, select a GPU runtime, and run cells top to bottom.
3. When asked to upload the project ZIP, upload the v4 ZIP, not the old ZIP.
4. Confirm the configuration prints `Training mode: all` and `Output folder: ...run_002`.
5. Run the training cell. The code clears only this designated `run_002` output folder before a fresh run so an old checkpoint cannot silently survive.
6. When completed, run the last notebook cell to download `AuthenticVision_Training_Results_run_002.zip`.
7. Copy the new `best_model.pt` to your Windows project's `outputs` folder, replacing the old checkpoint. Do not reuse `diagnostic_run/best_model.pt` or the old `run_001` checkpoint.

## Honest expectation

This is a better training protocol, not a guaranteed fix for every image. Tiny-GenImage has older and limited generator coverage compared with current AI tools. Test on a separate labelled collection representative of your real use. A few example images are enough for a smoke test but not enough to validate threshold/accuracy reliably.
