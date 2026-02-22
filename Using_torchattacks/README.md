# Using torchattacks Library

This directory contains scripts for generating and evaluating adversarial attacks using the official [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) library.

## Directory Structure

```
Using_torchattacks/
├── generate_adversarial_torchattacks.py  # Generate adversarial images
├── run_inference_torchattacks.py         # Run inference on adversarial images
├── README.md                              # This file
└── <ATTACK>/                              # Attack-specific results
    ├── adv_img_<params>/                  # Adversarial images
    ├── verification_adv_img_<params>/     # Verification images
    ├── results_adv_img_<params>.txt       # Inference results
    └── comparison_adv_img_<params>.png    # Comparison plot
```

## Installation

```bash
pip install torchattacks
```

## Usage

### Step 1: Generate Adversarial Images

**FGSM Attack:**
```bash
python Using_torchattacks/generate_adversarial_torchattacks.py --attack FGSM --eps 0.03
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack FGSM --eps 0.03
```

**PGD Attack:**
```bash
python Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start
```

**MIFGSM Attack:**
```bash
python Using_torchattacks/generate_adversarial_torchattacks.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
```

### Step 2: Run Inference

```bash
python Using_torchattacks/run_inference_torchattacks.py --attack <ATTACK> --folder <FOLDER_NAME>
```

**Example:**
```bash
# After generating FGSM with eps=0.03
python Using_torchattacks/run_inference_torchattacks.py --attack FGSM --folder adv_img_eps003
# py -3.9 Using_torchattacks/run_inference_torchattacks.py --attack FGSM --folder adv_img_eps003

# After generating PGD with eps=0.03, alpha=0.007, steps=10
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_randstart
# py -3.9 Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_randstart

#After generating MIFGSM with eps=0.03, alpha=0.007, steps=10, decay=1.0
python Using_torchattacks/run_inference_torchattacks.py --attack MIFGSM --folder adv_img_eps003_alpha0007_steps10_decay1
# py -3.9 Using_torchattacks/run_inference_torchattacks.py --attack MIFGSM --folder adv_img_eps003_alpha0007_steps10_decay1
```

## Parameters

### generate_adversarial_torchattacks.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--attack` | Attack method (FGSM, PGD, MIFGSM) | Required |
| `--eps` | Maximum perturbation | 8/255 (≈0.0314) |
| `--alpha` | Step size (PGD, MIFGSM) | 2/255 (≈0.0078) |
| `--steps` | Number of steps (PGD, MIFGSM) | 10 |
| `--decay` | Momentum decay (MIFGSM) | 1.0 |
| `--random_start` | Random initialization (PGD) | False |

### run_inference_torchattacks.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--attack` | Attack name | Required |
| `--folder` | Folder with adversarial images | Required |
| `--num_samples` | Number of samples to process | 200 |

## Output Files

### Adversarial Images
- Location: `Using_torchattacks/<ATTACK>/adv_img_<params>/`
- Format: Same filenames as original images (10550.jpg, 10551.jpg, ...)

### Verification Images
- Location: `Using_torchattacks/<ATTACK>/verification_adv_img_<params>/`
- Shows: Original, Adversarial, and Perturbation side-by-side
- Count: 10 images (first 5 + 5 random)

### Results File
- Location: `Using_torchattacks/<ATTACK>/results_adv_img_<params>.txt`
- Format:
  ```
  ImageName    GroundTruth    OriginalOutput    AdversarialOutput
  10550.jpg    0.0            -0.000023         0.007225
  ```

### Comparison Plot
- Location: `Using_torchattacks/<ATTACK>/comparison_adv_img_<params>.png`
- Shows: Ground Truth, Original Prediction, and Adversarial Output

## Differences from Custom Implementation

1. **Library**: Uses official torchattacks instead of custom attack classes
2. **Image Range**: Works with [0, 1] range (torchattacks convention)
3. **Regression Wrapper**: Includes `RegressionAttackWrapper` to adapt classification attacks for regression
4. **Folder Structure**: Separate `Using_torchattacks/` directory to keep organized

## Notes

- All attacks are adapted for **regression** (steering angle prediction)
- Model is set to **train mode** during attack generation for gradient computation
- Inference uses the same Lava SDNN pipeline as the original implementation
- Temporary files are automatically cleaned up after inference

## Comparison with Custom Implementation

You can compare results between:
- Custom attacks: `<ATTACK>/adv_img_<params>/`
- torchattacks: `Using_torchattacks/<ATTACK>/adv_img_<params>/`

Both should produce similar adversarial perturbations and attack effectiveness.