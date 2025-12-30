# Tiny-ImageNet + ResNet18 integration notes

This repository integrates Tiny-ImageNet and a torchvision-based ResNet18 backbone while keeping CLI semantics and output shapes consistent with the original continual-learning codebase.

## Changes after reviewing the reference setup
- **Backbone stem**: The `TorchVisionResNet` wrapper now uses a 3×3 stride-1 stem without max-pooling by default, mirroring Tiny-ImageNet-friendly configurations seen in the reference while avoiding the aggressive down-sampling that a 7×7 stride-2 + max-pool stem would introduce at 64×64 resolution.
- **Backbone aliases**: CLI options accept `resNet18`, `resnet18`, and `tv_resnet18`, and the model factory forwards these to the torchvision backbone even when `--depth 0` is used so the ResNet18 feature extractor is still chosen.
- **Layer metadata**: The layer-info helper reports the four residual stages `[64, 128, 256, 512]` matching the standard ResNet18 block layout `[2, 2, 2, 2]`, aligning with the reference description.
- **Tiny-ImageNet transforms**: Training augmentation matches the reference (random crop 64 with padding 4 + horizontal flip), while test/val splits avoid augmentation. Normalization uses the dataset mean/std documented alongside the reference pipeline.
- **Class ordering**: Tiny-ImageNet contexts keep a deterministic class order (no random permutation) so class-IL splits stay aligned with the 10×20 class partitioning described in the reference setting.

## How to run SI + adaptive regularization on Tiny-ImageNet with ResNet18
Example invocation consistent with the updated defaults and the reference backbone choice:

```bash
python main.py \
  --experiment TinyImageNet --scenario task --contexts 10 \
  --si --use-adaptive \
  --conv-type resNet18 --depth 0 --gp \
  --batch 256 --lr 0.0001 --iters 5000 \
  --augment --normalize
```

This selects the torchvision ResNet18 feature extractor, uses the Tiny-ImageNet transforms above, and activates SI + adaptive regularizers while keeping output units and task splits consistent with the existing codebase.
