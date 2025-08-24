# GI Tract Segmentation — U-Net vs SegFormer

End-to-end training, evaluation, and visualization pipelines for GI tract organ segmentation with two architectures:

- **U-Net** (from scratch, PyTorch)
- **SegFormer** (Transformer encoder via Hugging Face)

Includes a comprehensive evaluation suite (Dice, HD95, ASD, Surface-Dice, clDice, object-F1, volume similarity, FPR/FNR, continuity), an experiment summary CSV (`gi_seg_eval_summary.csv`), and a theoretical write-up (`segformer-cnn-segmentation-v4.pdf`) explaining when/why Transformers beat CNNs on tubular anatomy.

## TL;DR

Results from `gi_seg_eval_summary.csv` - see file for complete metrics.

### 📈 Quantitative Results (Small Bowel - Class 2)

| Metric               | Better | SegFormer | U-Net | Δ (S−U) |
|----------------------|:------:|---------:|------:|--------:|
| Dice                 | Higher | **0.877** | 0.820 | +0.058 |
| HD95 (px)            | Lower  | **5.914** | 11.177 | -5.263 |
| ASD (px)             | Lower  | **1.550** | 2.653 | -1.103 |
| Surface Dice @2 px   | Higher | **0.865** | 0.783 | +0.081 |
| Object-F1            | Higher | **0.842** | 0.724 | +0.117 |
| clDice               | Higher | **0.875** | 0.815 | +0.060 |

## Repository Contents

- 'GI-Tract-UNET.ipynb` — U-Net training/validation/inference + metrics
- `GI-Tract-segformer.ipynb` — SegFormer fine-tuning (HF SegformerForSemanticSegmentation) + metrics
- `gi_seg_eval_summary.csv` — consolidated experiment results
- `segformer-cnn-segmentation-v4.pdf` — theory & empirical alignment

Both training scripts:
- log to TensorBoard
- save top-k checkpoints
- dump JSON of eval stats
- export qualitative overlays (input / GT / prediction / overlay)

## Dataset & Labels

The scripts auto-download a prepared GI-tract dataset and unzip to:

```
./dataset_UWM_GI_Tract_train_valid/
  train/{images,masks}/*.png
  valid/{images,masks}/*.png
```

Default image size: 288×288 (configurable)

Class ids & colors:
- 0 Background (black)
- 1 Stomach (blue)
- 2 Small_Bowel (green)
- 3 Large_Bowel (red)

## Environment

Tested with Python 3.10+.

```bash
# PyTorch (choose CUDA/CPU wheels as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install lightning==2.* torchmetrics transformers albumentations opencv-python-headless
pip install scikit-image scipy matplotlib requests torchinfo
```

**Windows:** set `NUM_WORKERS=0`.  
**Linux/macOS:** workers auto-limited by script.



## Training Knobs You'll Likely Touch

- **Batch/Image size:** `TrainingConfig.BATCH_SIZE`, `DatasetConfig.IMAGE_SIZE`
- **LR/optimizer:** `TrainingConfig.INIT_LR`, `OPTIMIZER_NAME` (default AdamW), `WEIGHT_DECAY`
- **Augmentations:** see `MedicalDataset.setup_transforms(...)` (Albumentations)
- **Metrics:** `SegmentationMetrics` implements Dice, HD95, ASD, Surface-Dice @2 px, clDice, object-F1, volume similarity, FPR/FNR, continuity

## Outputs & Logging

```bash
tensorboard --logdir=tb_logs
```

**Artifacts:**
- `*_evaluation_results.json` — per-class summary with mean/std/median/quartiles
- `*/inference_results/*.png` — qualitative overlays
- `gi_seg_eval_summary.csv` — the consolidated results used in the TL;DR
- `segformer-cnn-segmentation-v4.pdf` — the theoretical write-up

## Repro Tips

- **Determinism:** `pl.seed_everything(42, workers=True)` in both scripts
- **Mixed precision:** enabled automatically on GPU (Lightning Trainer)
- **Dataloaders:** increase `NUM_WORKERS` and set `pin_memory=True` on Linux
- **VRAM:** if OOM with SegFormer, reduce batch size or image size

## Why Transformers Win Here (1-paragraph version)

Tubular anatomy (e.g., small bowel) demands global context to maintain topology and local precision for boundaries. Self-attention gives one-hop, non-local information flow along the structure, reducing breaks and false negatives, while SegFormer's decoder preserves edges. That's why, in the CSV, SegFormer improves Dice, HD95, ASD, Surface-Dice, clDice, and object-F1 over U-Net—precisely what the PDF formalizes.

## Citation

If you use this repo, please cite:

*Theoretical Framework for CNN vs Transformer Performance in Medical Segmentation: An Information-Geometric Analysis* — 2025.  
(see `segformer-cnn-segmentation-v4.pdf`)
