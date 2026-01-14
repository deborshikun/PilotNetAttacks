# Adversarial Attacks on a SDNN-PilotNet Model

This repository contains implementations of several **gradient-based adversarial attacks** specifically adapted for a **PilotNet-style Spiking Neural Network (SDNN)** that performs **steering angle prediction**. The attacks demonstrate the vulnerability of SDNNs to adversarial perturbations.

##  Project Structure

```
PilotNetAttacks/
│
├── Attacks/
│   └── attacks.py          #attack class definitions.
│
├── PGD/                    #attack files
├── FGSM/
├── MIFGSM/
.
.
.
├──pilotnet_sdnn/
│   ├── network.net         #.net file
│   ├── results.txt         #contains ground truth and original model predictions.
│   . 
│   .
│   .
└── testing_dataset/
    ├── 0.jpg               #input images.
    ├── 1.jpg
    └── ...
           
```

### Setup:
- Place your pre-trained SDNN model as `pilotnet_sdnn/network.pt`
- Place test images in `testing_dataset/`
- Place PyTorch model as `Trained/network.pt`

