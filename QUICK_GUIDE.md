# Adversarial Attack Pipeline - Quick Guide

### Step 1: Generate Adversarial Images
```bash
# Generate adversarial perturbations from testing_dataset
# Images will be saved in parameter-based folder names
python generate_adversarial_images.py --attack FGSM --eps 0.03
python generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python generate_adversarial_images.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0

# py -3.9 generate_adversarial_images.py --attack FGSM --eps 0.03
# py -3.9 generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
# py -3.9 generate_adversarial_images.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
```

**Output:** 
- `<ATTACK>/adv_img_eps0.03/` - For FGSM
- `<ATTACK>/adv_img_eps0.03_alpha0.007_steps10/` - For PGD
- `<ATTACK>/adv_img_eps0.03_alpha0.007_steps10_decay1.0/` - For MIFGSM etc.

### Step 2: Run Inference on Adversarial Images
```bash
# Run Lava SDNN inference on adversarial images
# Use --folder to specify which adversarial images to process
python run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps0.03
python run_inference_on_adversarial.py --attack PGD --folder adv_img_eps0.03_alpha0.007_steps10
python run_inference_on_adversarial.py --attack MIFGSM --folder adv_img_eps0.03_alpha0.007_steps10_decay1.0

py -3.9 run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps0.03
py -3.9 run_inference_on_adversarial.py --attack PGD --folder adv_img_eps0.03_alpha0.007_steps10
py -3.9 run_inference_on_adversarial.py --attack MIFGSM --folder adv_img_eps0.03_alpha0.007_steps10_decay1.0
```

**Output:**
- `<ATTACK>/results_<folder_name>.txt` - Complete results comparing original vs adversarial predictions
- `<ATTACK>/comparison_<folder_name>.png` - Visualization plot with 3 lines

## Results

The `results_<ATTACK>.txt` file contains:
```
ImageName     GroundTruth    OriginalOutput    AdversarialOutput
10550.jpg     0.0            -0.000023         0.007225
10551.jpg     0.078278       -0.000023         0.007225
...
```

### `generate_adversarial_images.py` ::
- Loads PyTorch model (`Trained/network.pt`)
- Loads images from `testing_dataset/`
- Applies attack from `Attacks/attacks.py`
- Saves perturbed images to `<ATTACK>/adv_img_<parameters>/`
  - Folder name includes attack parameters (e.g., `adv_img_eps0.03`)

### `run_inference_on_adversarial.py` ::
- Uses Lava network (`pilotnet_sdnn/network.net`)
- Reads original predictions (`pilotnet_sdnn/results.txt`)
- Runs inference on adversarial images from specified folder
- Compares original vs adversarial predictions
- Generates results file and comparison plot

## 📁 Directory Structure
```
PilotNetAttacks/
├── testing_dataset/           # Your test images (0.jpg - 200.jpg)
├── pilotnet_sdnn/
│   ├── network.net           # Lava SDNN model
│   └── results.txt           # Original predictions
├── Trained/
│   └── network.pt            # PyTorch model for attacks
├── Attacks/
│   └── attacks.py            # FGSM, PGD, MIFGSM implementations
├── FGSM/
│   ├── adv_img_eps0.03/      # FGSM perturbed images
│   ├── results_adv_img_eps0.03.txt
│   └── comparison_adv_img_eps0.03.png
├── PGD/
│   ├── adv_img_eps0.03_alpha0.007_steps10/
│   ├── results_adv_img_eps0.03_alpha0.007_steps10.txt
│   └── comparison_adv_img_eps0.03_alpha0.007_steps10.png
└── MIFGSM/
    ├── adv_img_eps0.03_alpha0.007_steps10_decay1.0/
    ├── results_adv_img_eps0.03_alpha0.007_steps10_decay1.0.txt
    └── comparison_adv_img_eps0.03_alpha0.007_steps10_decay1.0.png
```

## ✨ Example Usage

```bash
# Run complete pipeline for PGD attack
python generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python run_inference_on_adversarial.py --attack PGD --folder adv_img_eps0.03_alpha0.007_steps10

# Check results
cat PGD/results_adv_img_eps0.03_alpha0.007_steps10.txt
# View plot: PGD/comparison_adv_img_eps0.03_alpha0.007_steps10.png
```

## 🔧 Customization

### Attack Parameters
- `--eps`: Perturbation magnitude (0.01 - 0.1)
- `--alpha`: Step size for iterative attacks
- `--steps`: Number of iterations (more = stronger)
- `--decay`: Momentum for MIFGSM

**Note:** Folder names automatically reflect these parameters

### Inference Parameters
- `--folder`: Name of the adversarial images folder (required)
- `--num_samples`: Process fewer samples (default: 200)

## 📈 Success Metrics

A successful attack will show:
- **MSE Increase:** > 100% (higher is more effective)
- **Adversarial predictions** differ significantly from original
- **Visual perturbations** are minimal (low average perturbation value)

## ✅ Verification

After running the pipeline, you should have:
1. Adversarial images in `<ATTACK>/adv_img_<parameters>/`
2. Results file with 4 columns in `<ATTACK>/results_adv_img_<parameters>.txt`
3. Comparison plot in `<ATTACK>/comparison_adv_img_<parameters>.png`
4. Console output showing MSE increase percentage
