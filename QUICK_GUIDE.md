# Adversarial Attack Pipeline - Quick Guide

## Two Attack Approaches

### Approach 1: Full Sequence Attack (Original)
Attacks all 200 frames as one sequence - faster but less thorough.

### Approach 2: Sliding Window Attack (Recommended by Professors)
Attacks overlapping 5-frame windows independently - more thorough testing.

---

## Step 1: Generate Adversarial Images

### Option A: Full Sequence Attack (Custom Implementation)
```bash
# Generate adversarial perturbations from testing_dataset
# Images will be saved in parameter-based folder names
python generate_adversarial_images.py --attack FGSM --eps 0.03
python generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python generate_adversarial_images.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0

# Windows:
# py -3.9 generate_adversarial_images.py --attack FGSM --eps 0.03
# py -3.9 generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
# py -3.9 generate_adversarial_images.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
```

**Output:** 
- `<ATTACK>/adv_img_eps003/` - For FGSM
- `<ATTACK>/adv_img_eps003_alpha0007_steps10/` - For PGD
- `<ATTACK>/adv_img_eps003_alpha0007_steps10_decay1/` - For MIFGSM

### Option B: Full Sequence Attack (torchattacks Library)
```bash
# Using torchattacks library wrapper
python Using_torchattacks/generate_adversarial_torchattacks.py --attack FGSM --eps 0.03
python Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start
python Using_torchattacks/generate_adversarial_torchattacks.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0

# Windows:
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack FGSM --eps 0.03
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start
# py -3.9 Using_torchattacks/generate_adversarial_torchattacks.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
```

**Output:**
- `Using_torchattacks/<ATTACK>/adv_img_eps003/`
- `Using_torchattacks/<ATTACK>/adv_img_eps003_alpha0007_steps10_randstart/`
- `Using_torchattacks/<ATTACK>/adv_img_eps003_alpha0007_steps10_decay1/`

### Option C: Sliding Window Attack (Recommended - Most Thorough)
```bash
# Attacks overlapping 5-frame windows independently (196 separate attacks for 200 frames)
# ⚠️ Warning: Much slower (~30-60 minutes) but more thorough testing
python Using_torchattacks/generate_adversarial_sliding_window.py --attack FGSM --eps 0.03 --window_size 5
python Using_torchattacks/generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5
python Using_torchattacks/generate_adversarial_sliding_window.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0 --window_size 5

# Windows:
# py -3.9 Using_torchattacks/generate_adversarial_sliding_window.py --attack FGSM --eps 0.03 --window_size 5
# py -3.9 Using_torchattacks/generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5
# py -3.9 Using_torchattacks/generate_adversarial_sliding_window.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0 --window_size 5
```

**Output:**
- `Using_torchattacks/<ATTACK>/adv_img_eps003_win5/`
- `Using_torchattacks/<ATTACK>/adv_img_eps003_alpha0007_steps10_win5/`
- `Using_torchattacks/<ATTACK>/adv_img_eps003_alpha0007_steps10_decay1_win5/`

**Note:** The `_win5` suffix indicates sliding window attack with window size 5.

### Option D: Single Frame Impact Analysis (Growing Window)
```bash
# STEP 1: Generate adversarial frames with temporal context
# Each frame i is perturbed within context [f0...fi]
# ⚠️ Warning: Slow (~30-45 minutes) - generates 197 separate adversarial frames
python Using_torchattacks/sa_compare/generate_single_frame_adversarial.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python Using_torchattacks/sa_compare/generate_single_frame_adversarial.py --attack FGSM --eps 0.03
python Using_torchattacks/sa_compare/generate_single_frame_adversarial.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0

# Windows:
# py -3.9 Using_torchattacks/sa_compare/generate_single_frame_adversarial.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10

# STEP 2: Analyze impact using growing windows (FAST - reusable)
# Uses pre-generated adversarial frames for windowed analysis
python Using_torchattacks/sa_compare/analyze_single_frame_impact.py --attack PGD --folder single_frame_eps003_alpha0007_steps10
python Using_torchattacks/sa_compare/analyze_single_frame_impact.py --attack FGSM --folder single_frame_eps003
python Using_torchattacks/sa_compare/analyze_single_frame_impact.py --attack MIFGSM --folder single_frame_eps003_alpha0007_steps10_decay1

# Windows:
# py -3.9 Using_torchattacks/sa_compare/analyze_single_frame_impact.py --attack PGD --folder single_frame_eps003_alpha0007_steps10
```

**Output (Step 1):**
- `Using_torchattacks/sa_compare/<ATTACK>/adv_images_single_frame_<params>/` - Generated frames
- `Using_torchattacks/sa_compare/<ATTACK>/verification_single_frame_<params>/` - Verification images

**Output (Step 2):**
- `Using_torchattacks/sa_compare/<ATTACK>/results_single_frame_<params>.txt` - Detailed results
- `Using_torchattacks/sa_compare/<ATTACK>/results_single_frame_<params>.csv` - Excel-ready data
- `Using_torchattacks/sa_compare/<ATTACK>/checking_single_frame_<params>/` - Window verification images

**What this does:**
- **Step 1 (Slow):** Generates adversarial version of each frame perturbed within its temporal context
  - Frame 0: Perturbed in context [f0]
  - Frame 1: Perturbed in context [f0, f1]
  - Frame i: Perturbed in context [f0, f1, ..., fi]
- **Step 2 (Fast):** Analyzes incremental impact using growing windows
  - Iteration 1: `[f0 clean]` vs `[f0 perturbed]`
  - Iteration 2: `[f0, f1 clean]` vs `[f0 clean, f1 perturbed]`
  - Iteration i: `[f0...fi clean]` vs `[f0...f(i-1) clean, fi perturbed]`
- Shows which individual frames have the most impact on steering decisions

---

## Step 2: Run Inference on Adversarial Images

### For Custom Implementation (Option A)
```bash
# Run Lava SDNN inference on adversarial images
python run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps003
python run_inference_on_adversarial.py --attack PGD --folder adv_img_eps003_alpha0007_steps10
python run_inference_on_adversarial.py --attack MIFGSM --folder adv_img_eps003_alpha0007_steps10_decay1

# Windows:
# py -3.9 run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps003
# py -3.9 run_inference_on_adversarial.py --attack PGD --folder adv_img_eps003_alpha0007_steps10
# py -3.9 run_inference_on_adversarial.py --attack MIFGSM --folder adv_img_eps003_alpha0007_steps10_decay1
```

### For torchattacks Implementation (Options B & C)
```bash
# Run inference on torchattacks-generated adversarial images
python Using_torchattacks/run_inference_torchattacks.py --attack FGSM --folder adv_img_eps003
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_randstart
python Using_torchattacks/run_inference_torchattacks.py --attack MIFGSM --folder adv_img_eps003_alpha0007_steps10_decay1

# For sliding window results (add _win5):
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_win5

# Windows:
# py -3.9 Using_torchattacks/run_inference_torchattacks.py --attack FGSM --folder adv_img_eps003
# py -3.9 Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_win5
```

**Output:**
- `Using_torchattacks/<ATTACK>/results_<folder_name>.txt` - Complete results
- `Using_torchattacks/<ATTACK>/comparison_<folder_name>.png` - Visualization plot

---

---

## Comparison of Approaches

| Approach | Attacks | Speed | Thoroughness | Use Case |
|----------|---------|-------|--------------|----------|
| **Full Sequence (Option A/B)** | 1 attack on 200 frames | Fast (~10 sec) | Basic | Quick testing |
| **Sliding Window (Option C)** | 196 attacks (overlapping 5-frame windows) | Slow (~30-60 min) | Thorough | Research/publication |
| **Single Frame (Option D)** | 197 individual frame perturbations | Step 1: ~30-45 min<br>Step 2: ~1-2 min | Per-frame impact | Individual frame impact analysis |

**Professor's Recommendation:** Use Option C (Sliding Window) for thorough testing and research purposes. Use Option D to identify which specific frames have the most impact on steering predictions.

---

## Results Format

### Full Sequence Attacks (Options A & B)
The `results_<ATTACK>.txt` file contains:
```
ImageName     GroundTruth    OriginalOutput    AdversarialOutput
10550.jpg     0.0            -0.000023         0.007225
10551.jpg     0.078278       -0.000023         0.007225
...
```

### Single Frame Impact Analysis (Option D)
The `results_single_frame_<params>.txt` file contains:
```
Frame 0:
  Clean sequence [0]: predicted = -0.000123
  Adversarial sequence [0*]: predicted = 0.007225
  Steering difference: 0.007348

Frame 1:
  Clean sequence [0, 1]: predicted = -0.000156
  Adversarial sequence [0, 1*]: predicted = 0.008134
  Steering difference: 0.008290
...
```
Where `*` indicates the perturbed frame. Shows incremental impact of each individual frame on final steering output.

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
├── testing_dataset/           # Your test images (0.jpg - 199.jpg)
├── pilotnet_sdnn/
│   ├── network.net           # Lava SDNN model (inference)
│   └── results.txt           # Original predictions
├── Trained/
│   └── network.pt            # PyTorch model (for attacks)
├── Attacks/
│   └── attacks.py            # Custom FGSM, PGD, MIFGSM implementations
├── FGSM/                      # Custom implementation results
│   ├── adv_img_eps003/
│   ├── results_adv_img_eps003.txt
│   └── comparison_adv_img_eps003.png
├── PGD/                       # Custom implementation results
│   ├── adv_img_eps003_alpha0007_steps10/
│   ├── results_adv_img_eps003_alpha0007_steps10.txt
│   └── comparison_adv_img_eps003_alpha0007_steps10.png
├── MIFGSM/                    # Custom implementation results
│   ├── adv_img_eps003_alpha0007_steps10_decay1/
│   ├── results_adv_img_eps003_alpha0007_steps10_decay1.txt
│   └── comparison_adv_img_eps003_alpha0007_steps10_decay1.png
└── Using_torchattacks/        # torchattacks library results
    ├── generate_adversarial_torchattacks.py
    ├── generate_adversarial_sliding_window.py  # Sliding window implementation
    ├── run_inference_torchattacks.py
    ├── sa_compare/                # Single frame analysis scripts
    │   ├── generate_single_frame_adversarial.py
    │   ├── analyze_single_frame_impact.py
    │   ├── PGD/
    │   │   ├── adv_images_single_frame_eps003_alpha0007_steps10/  # Generated frames
    │   │   ├── verification_single_frame_eps003_alpha0007_steps10/
    │   │   ├── checking_single_frame_eps003_alpha0007_steps10/    # Analysis images
    │   │   ├── results_single_frame_eps003_alpha0007_steps10.txt
    │   │   └── results_single_frame_eps003_alpha0007_steps10.csv
    │   ├── FGSM/
    │   └── MIFGSM/
    ├── FGSM/
    │   ├── adv_img_eps003/           # Full sequence attack
    │   ├── adv_img_eps003_win5/      # Sliding window attack
    │   └── results_*.txt
    ├── PGD/
    │   ├── adv_img_eps003_alpha0007_steps10/
    │   ├── adv_img_eps003_alpha0007_steps10_win5/  # Sliding window
    │   ├── results_*.txt
    │   └── comparison_*.png
    └── MIFGSM/
        ├── adv_img_eps003_alpha0007_steps10_decay1/
        ├── adv_img_eps003_alpha0007_steps10_decay1_win5/  # Sliding window
        └── results_*.txt
```

## ✨ Example Usage

### Quick Test (Full Sequence - Fast)
```bash
# Run complete pipeline for PGD attack (takes ~10 seconds)
python generate_adversarial_images.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python run_inference_on_adversarial.py --attack PGD --folder adv_img_eps003_alpha0007_steps10

# Check results
cat PGD/results_adv_img_eps003_alpha0007_steps10.txt
# View plot: PGD/comparison_adv_img_eps003_alpha0007_steps10.png
```

### Thorough Testing (Sliding Window - Recommended)
```bash
# Run complete pipeline with sliding window approach (takes ~30-60 minutes)
python Using_torchattacks/generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_win5

# Check results
cat Using_torchattacks/PGD/results_adv_img_eps003_alpha0007_steps10_win5.txt
cat Using_torchattacks/PGD/window_statistics.txt  # Detailed per-window stats
# View plot: Using_torchattacks/PGD/comparison_adv_img_eps003_alpha0007_steps10_win5.png
```

### Compare Both Approaches
```bash
# Generate with both methods
python Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python Using_torchattacks/generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5

# Run inference on both
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_win5

# Compare MSE increase between methods
```

## 🔧 Customization

### Attack Parameters
- `--eps`: Perturbation magnitude (e.g., 0.01, 0.03, 0.1)
- `--alpha`: Step size for iterative attacks (typically eps/4 to eps/2)
- `--steps`: Number of iterations (more = stronger, 10-40 typical)
- `--decay`: Momentum for MIFGSM (0.0 to 1.0, default 1.0)
- `--random_start`: Add random initialization for PGD (flag)
- `--window_size`: Temporal window size for sliding window attack (default: 5)

**Note:** Folder names automatically reflect these parameters

### Choosing Window Size
- **window_size=5**: Default, balances thoroughness and speed (196 attacks for 200 frames)
- **window_size=3**: Faster, more attacks (198 windows) but smaller context
- **window_size=10**: Slower, fewer attacks (191 windows) but larger temporal context

### Inference Parameters
- `--folder`: Name of the adversarial images folder (required, must match generation output)
- `--num_samples`: Process fewer samples (default: 200)

## 📈 Success Metrics

A successful attack will show:
- **MSE Increase:** > 100% (higher is more effective)
- **Adversarial predictions** differ significantly from original
- **Visual perturbations** are minimal (low average perturbation value)

**Note:** Sliding window attacks typically show higher MSE increase due to more thorough testing.

---

## 🚀 Quick Reference Commands

### Fastest (10 seconds):
```bash
python generate_adversarial_images.py --attack FGSM --eps 0.03
python run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps003
```

### Most Thorough (30-60 minutes - Recommended for Research):
```bash
python Using_torchattacks/generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10_win5
```

### Standard (balanced):
```bash
python Using_torchattacks/generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
python Using_torchattacks/run_inference_torchattacks.py --attack PGD --folder adv_img_eps003_alpha0007_steps10
```

### Single Frame Impact Analysis (2-phase workflow):
```bash
# Phase 1: Generate (slow, run once - 30-45 minutes)
python Using_torchattacks/sa_compare/generate_single_frame_adversarial.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10

# Phase 2: Analyze (fast, reusable - 1-2 minutes)
python Using_torchattacks/sa_compare/analyze_single_frame_impact.py --attack PGD --folder single_frame_eps003_alpha0007_steps10
```

---

## 📚 Additional Resources

- **FAQ.md**: Detailed explanations of technical decisions and common questions
- **Using_torchattacks/SLIDING_WINDOW_EXPLANATION.md**: Visual guide to sliding window approach
- **Attacks/implementation.txt**: Notes on custom attack implementations

## ✅ Verification

After running the pipeline, you should have:

### For Full Sequence Attacks (Options A & B):
1. Adversarial images in `<ATTACK>/adv_img_<parameters>/`
2. Results file with 4 columns in `<ATTACK>/results_adv_img_<parameters>.txt`
3. Comparison plot in `<ATTACK>/comparison_adv_img_<parameters>.png`
4. Console output showing MSE increase percentage

### For Sliding Window Attacks (Option C):
1. Adversarial images in `Using_torchattacks/<ATTACK>/adv_img_<parameters>_win<N>/`
2. Results file in `Using_torchattacks/<ATTACK>/results_adv_img_<parameters>_win<N>.txt`
3. **Window statistics** in `Using_torchattacks/<ATTACK>/adv_img_<parameters>_win<N>/window_statistics.txt`
4. Comparison plot in `Using_torchattacks/<ATTACK>/comparison_adv_img_<parameters>_win<N>.png`
5. Verification images for first 10, middle 10, and last 10 frames
6. Console output showing per-window progress and final MSE increase

### For Single Frame Impact Analysis (Option D):
**After Step 1 (Generation):**
1. Adversarial frames in `Using_torchattacks/sa_compare/<ATTACK>/adv_images_single_frame_<parameters>/`
2. Verification images in `Using_torchattacks/sa_compare/<ATTACK>/verification_single_frame_<parameters>/`
3. Console showing "Generated adversarial version for frame X/197"

**After Step 2 (Analysis):**
1. Results file in `Using_torchattacks/sa_compare/<ATTACK>/results_single_frame_<parameters>.txt`
2. CSV export in `Using_torchattacks/sa_compare/<ATTACK>/results_single_frame_<parameters>.csv`
3. Checking images in `Using_torchattacks/sa_compare/<ATTACK>/checking_single_frame_<parameters>/`
4. Console showing per-frame clean vs adversarial predictions and steering differences

### Expected Window Statistics Format:
```
Sliding Window Attack Statistics
======================================================================
Attack: PGD
Window size: 5
Epsilon: 0.03
Number of windows: 196

Window | Frames      | Target Value | Perturbation Norm
----------------------------------------------------------------------
     0 | [  0,   4] |    -0.248081 |           0.015234
     1 | [  1,   5] |    -0.251203 |           0.014891
     2 | [  2,   6] |    -0.249456 |           0.015102
...
```
