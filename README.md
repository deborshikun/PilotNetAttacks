# Adversarial Attacks on a SDNN-PilotNet Model

This repository contains implementations of several **gradient-based adversarial attacks** specifically adapted for a **PilotNet-style Spiking Neural Network (SDNN)** that performs **steering angle prediction**. The attacks demonstrate the vulnerability of SDNNs to adversarial perturbations.

##  Project Structure

```
PilotNetAttacks/
│
├── Attacks/
│   └── attacks.py          # Core file with all attack class definitions.
│
├── FGSM/
│   ├── modular_txt.py      # Main script to run FGSM and generate text report.
│   └── modular_plot.py     # Main script to run FGSM and generate plot.
│
├── PGD/
│   ├── modular_txt.py      # Main script to run PGD and generate text report.
│   └── modular_plot.py     # Main script to run PGD and generate plot.
│
├── MIFGSM/
│   ├── modular_txt.py      # Main script to run MIFGSM and generate text report.
│   └── modular_plot.py     # Main script to run MIFGSM and generate plot.
│
├── Trained/
│   └── network.pt          # Pre-trained SNN model goes here.
│
├── testing_dataset/
│   ├── 0.jpg               # Input images.
│   ├── 1.jpg
│   └── ...
│
└── results.txt             # Contains ground truth and original model predictions.
```

---

## Installation

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd PilotNetAttacks
```

### 3. Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Setup:
- Place your pre-trained SNN model as `Trained/network.pt`
- Place test images in `testing_dataset/`
- Ensure `results.txt` is present in the root folder with format:
  
  ```
  ImageName   GroundTruth   Output   Difference
  ```

---

### Running an Attack

#### Example 1: PGD Attack with Text Report
```bash
cd PGD/
python modular_txt.py
```
Output: `PGD/Results/adversarial_comparison.txt`

#### Example 2: FGSM Attack with Plot
```bash
cd FGSM/
python modular_plot.py
```
Output: `FGSM/Results/attack_comparison.png`
