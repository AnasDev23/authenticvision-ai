# Research notes and evidence

## Dataset selection

The starter pipeline uses `TheKernel01/Tiny-GenImage` on Hugging Face. The dataset card lists:

- 35,000 rows total
- 28,000 train rows and 7,000 validation rows
- 8.36 GB total file size
- binary label: `0 real`, `1 fake`
- generator metadata covering real plus multiple generators
- license `CC-BY-NC-SA-4.0`

Source: <https://huggingface.co/datasets/TheKernel01/Tiny-GenImage>

This is a manageable starter subset in the GenImage family, not the original full one-million-pair benchmark. The official GenImage GitHub describes over one million AI/real pairs, generator folders such as Midjourney, VQDM, Wukong, Stable Diffusion variants, GLIDE, BigGAN and ADM, and the standard `train/ai`, `train/nature`, `val/ai`, `val/nature` layout.

Source: <https://github.com/GenImage-Dataset/GenImage>

## Cloud GPU notes

Google Colab says it offers access to GPUs/TPUs but that resources are not guaranteed and usage limits fluctuate. Therefore the notebook checks for a GPU and saves checkpoints to mounted Google Drive after each epoch.

Source: <https://research.google.com/colaboratory/faq.html>

## Model strategy

The project fine-tunes torchvision's ImageNet-pretrained ConvNeXt-Tiny and replaces its final layer with two classes. This is a practical transfer-learning baseline. It is not an exact reimplementation of Smaug and does not claim the same benchmark performance. `best_model.pt` is learned from this project's run.

## Evaluation strategy

- Exclude one AI generator (Midjourney by default) from training.
- Use a validation subset containing real images reserved from the train partition and AI images from non-held-out generators in the validation partition.
- Calibrate threshold against validation real-image scores with a target 5% empirical false-positive rate.
- Report a separate held-out generator test using the chosen generator's AI images and separate real examples from the validation partition.

The holdout set is small. Reported metrics describe only this data/split and should not be generalized into a guarantee for real-world images.

## Legal/usage note

Tiny-GenImage's displayed license is CC-BY-NC-SA-4.0. Users must check dataset terms and the licenses of constituent AI generators/source images before redistributing data, weights derived from it, or commercializing a service. The MIT license in this archive applies only to the project's code.


## Corrected deployment training (v4)

The first run's `Midjourney` holdout model was a research evaluation checkpoint and should not have been presented as the recommended deployment checkpoint for arbitrary user images. It excluded Midjourney fakes from training by design. The corrected deployment notebook uses `HOLDOUT_GENERATOR='all'`, includes all generator families present in the training split, and uses a disjoint image-level validation/test split from the official validation partition. This changes the claim: the new test measures performance on different images from generator families represented during training; it is not an unseen-generator test.

The v4 training run also uses a lower learning rate for the pretrained backbone than for its classifier head, mild JPEG compression augmentation, label smoothing, and a validation-loss plateau scheduler. These are reasonable changes to test generalization and overconfidence; they do not guarantee that every unknown real/AI pair will be correctly distinguished. Maintain a separate, labelled external test set for user-relevant images.
