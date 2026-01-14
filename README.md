# Adversarial Attacks on a SDNN-PilotNet Model

This repository contains implementations of several **gradient-based adversarial attacks** specifically adapted for a **PilotNet-style Spiking Neural Network (SDNN)** that performs **steering angle prediction**. The attacks demonstrate the vulnerability of SDNNs to adversarial perturbations.

##  Project Structure

```
PilotNetAttacks/
│
├── Attacks/
│   └── attacks.py          #attack class definitions.
│
├── testing_dataset/
│   ├── 0.jpg               # Input images.
│   ├── 1.jpg
│   └── ...
│
└── results.txt             # Contains ground truth and original model predictions.
```

### Setup:
- Place your pre-trained SDNN model as `pilotnet_sdnn/network.pt`
- Place test images in `testing_dataset/`
- Place PyTorch model as `Trained/network.pt`

