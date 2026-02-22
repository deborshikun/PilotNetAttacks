# Single Frame Perturbation Analysis - Growing Window Approach

This directory contains tools for analyzing the incremental impact of each frame as it's added to the temporal sequence.

## Purpose

Instead of perturbing all 200 frames simultaneously, this analysis uses a **growing window approach** where each iteration adds one more frame and compares clean vs perturbed versions. This reveals:
- How much each new frame affects the prediction when added to the sequence
- The incremental contribution of each frame
- Which frames cause the largest deviations when perturbed

## How It Works

For each iteration i (from 1 to 200):
1. **Run inference on clean window** [f0, f1, ..., fi all clean] → get SA_clean
2. **Run inference on perturbed window** [f0, f1, ..., f(i-1) clean, fi perturbed] → get SA_perturbed
3. **Compare the two steering angles** to see impact of perturbing frame i

Example progression:
```
Iteration 1:
  Clean:     [f0]           → SA = 0.123
  Perturbed: [f0-PERT]      → SA = 0.145
  Impact: |0.145 - 0.123| = 0.022

Iteration 2:
  Clean:     [f0, f1]       → SA = 0.125
  Perturbed: [f0, f1-PERT]  → SA = 0.150
  Impact: |0.150 - 0.125| = 0.025

Iteration 3:
  Clean:     [f0, f1, f2]       → SA = 0.127
  Perturbed: [f0, f1, f2-PERT]  → SA = 0.148
  Impact: |0.148 - 0.127| = 0.021

...and so on until iteration 200
```

Each iteration isolates the impact of the newly added frame in its temporal context.

## Usage

### Basic Usage (PGD with default parameters)

**Linux/Mac:**
```bash
cd Using_torchattacks/sa_compare
python single_frame_perturbation_analysis.py --attack PGD
```

**Windows (Python 3.9):**
```powershell
cd Using_torchattacks\sa_compare
py -3.9 single_frame_perturbation_analysis.py --attack PGD
```

### Custom Parameters

**FGSM:**
```bash
# Linux/Mac
python single_frame_perturbation_analysis.py --attack FGSM --eps 0.03

# Windows
py -3.9 single_frame_perturbation_analysis.py --attack FGSM --eps 0.03
```

**PGD:**
```bash
# Linux/Mac
python single_frame_perturbation_analysis.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10

# Windows
py -3.9 single_frame_perturbation_analysis.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10
```

**PGD with Random Start:**
```bash
# Linux/Mac
python single_frame_perturbation_analysis.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start

# Windows
py -3.9 single_frame_perturbation_analysis.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --random_start
```

**MIFGSM:**
```bash
# Linux/Mac
python single_frame_perturbation_analysis.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0

# Windows
py -3.9 single_frame_perturbation_analysis.py --attack MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
```

## Output

### Directory Structure
```
sa_compare/
├── PGD/
│   ├── results_single_frame_eps003_alpha0007_steps10.txt
│   └── adv_images_single_frame_eps003_alpha0007_steps10/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
├── FGSM/
│   ├── results_single_frame_eps003.txt
│   └── adv_images_single_frame_eps003/
└── MIFGSM/
    ├── results_single_frame_eps003_alpha0007_steps10_decay1.txt
    └── adv_images_single_frame_eps003_alpha0007_steps10_decay1/
```

### Results File Format

The results file contains:

1. **Configuration**: Attack parameters used
2. **Approach Description**: Growing window method
3. **Per-Iteration Table**: 
   - Iteration number (1-200)
   - Frame number (0-199)
   - Window size (how many frames in the sequence)
   - Image filename
   - Clean steering angle (for clean window)
   - Perturbed steering angle (with frame i perturbed)
   - Absolute difference
   - Percentage difference
4. **Summary Statistics**: Mean, max, min, std dev of absolute differences
5. **Top 10 Most Impactful Frames**: Frames that cause the largest changes when perturbed

Example table:
```
Iter  Frame   Window    Image     Clean SA    Perturbed SA   Abs Diff    % Diff
--------------------------------------------------------------------------------------
1     0       1         0.jpg     0.123456    0.125789       0.002333    1.89
2     1       2         1.jpg     0.125000    0.121234      -0.003766   -3.01
3     2       3         2.jpg     0.127000    0.149000       0.022000   17.32
...
```

**CSV File**: A CSV file is also generahas strong impact when added to sequence
- **Low absolute difference** = Frame has minimal incremental impact
- **Early iterations** show how much each of the first few frames matter
- **Later iterations** reveal how additional frames affect already-established predictions
- **Temporal accumulation** effects become visible (e.g., how perturbations in later frames compound with clean earlier frames)

## Compared to Full Sequence Attack

| Method                          | What It Tests                                    | Approach              |
|---------------------------------|--------------------------------------------------|-----------------------|
| **Full Sequence Attack**        | Overall robustness (all 200 frames perturbed)   | Static, all at once   |
| **Growing Window (this)**       | Incremental frame contribution, temporal buildup | Dynamic, one at a time|

This analysis is complementary to full sequence attacks - it provides temporal granularity showing how each frame contributes as the sequence builds
| **Full Sequence Attack**        | Overall robustness (all frames perturbed)| 200 (all)     |
| **Single Frame Analysis (this)**| Individual frame importance              | 1 per test    |

This analysis is complementary to full sequence attacks - it provides frame-level granularity.

## Processing Time
60-90 minutes for 200 frames
- **Per Iteration**: ~20-30 seconds (attack generation + 2 inferences)
- **Total Inferences**: 400 (2 per iteration × 200 iterations)
- **Total Attacks Generated**: 200 (one per frame
- **Total Inferences**: 201 (1 baseline + 200 perturbed sequences)

## Example Workflow

1. **Run the analysis:**
   ```bash
   py -3.9 single_frame_perturbation_analysis.py --attack PGD --eps 0.03
   ```

2. **Check results:**
   # Or open the CSV file in Excel
   start PGD/results_single_frame_eps003_alpha0007_steps10.csv  # Windows
   ```

3. **Analyze patterns:**
   - Look at "Top 10 Most Impactful Frames" section
   - Check how impact changes as window grows
   - Import CSV into Excel for plotting and deeper analysis

4. **Create visualizations (in Excel or Python):**
   - Plot: Iteration (x-axis) vs Absolute Difference (y-axis)
   - Shows which frames cause largest deviations when perturbed
   - Reveals temporal patterns in frame importance
   - Run full sequence attack: `py -3.9 ../generate_adversarial_torchattacks.py --attack PGD --eps 0.03`
   - Compare overall MSE increase vs per-frame impacts

## Tips

- **Start with PGD** (balanced effectiveness and speed)
- **Use consistent parameters** with your full sequence attacks for fair comparison
- **Focus on top impactful frames** for deeper analysis
- **Compare different attacks** to see if frame importance changes with attack method
