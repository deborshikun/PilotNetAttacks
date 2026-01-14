# Frequently Asked Questions - Adversarial Attacks

## Q1: Why use `model.train()` instead of `model.eval()` when generating adversarial images?

**Short Answer:** We need `model.train()` to enable gradient computation through the network during the attack generation phase. This does NOT retrain the model or change any weights.

**Detailed Explanation:**

`model.train()` does **NOT** retrain the model - it just changes the mode for how layers behave during forward pass. No weights are updated.

**`model.train()` - Training Mode:**
- Dropout layers are **active** (randomly drop neurons)
- BatchNorm uses **batch statistics** (not running averages)
- **Gradients can flow** through all layers
- **Weights are NOT updated** unless you call `optimizer.step()`

**`model.eval()` - Evaluation Mode:**
- Dropout layers are **disabled** (all neurons active)
- BatchNorm uses **running statistics** (frozen)
- Some layers may **block gradients**
- Used for inference only

**For adversarial attacks:**
- We need `model.train()` so gradients flow properly
- But we never call `optimizer.step()`, so **weights stay frozen**
- We only compute gradients w.r.t. **input images**, not model parameters

Your model weights remain exactly the same - we're just enabling gradient computation through the network architecture. The attack perturbs the **input**, not the **model**.

---

## Q2: I perturb the images first independently of the model, so why does model mode matter?

**The Misconception:**
It seems like the pipeline should be:
1. Perturb images (independent of model)
2. Run inference on perturbed images

**The Reality:**
Adversarial perturbations are **not** independent of the model! They are specifically calculated using the model's gradients.

**How FGSM Actually Works:**

1. **Input image** → Model → **Output prediction**
2. Calculate **loss** = (prediction - target)²
3. Compute **gradient of loss w.r.t. INPUT image** using backpropagation through the model
4. Perturb image: `adversarial = original + ε × sign(gradient)`

**The key insight:** To create the perturbation, we need to **backpropagate through the model** to get gradients at the input layer. This requires:
- Forward pass through model (to get prediction)
- Backward pass through model (to get input gradients)

**Why model.train() matters:**

When you call the attack in Step 1 of your pipeline:
```python
adv_sequence = attack(sequence_tensor, target)
```

Inside the attack:
```python
outputs, _, _ = self.model(images)  # Forward pass
cost = self.loss(final_prediction, target)
grad = torch.autograd.grad(cost, images, ...)  # Backward pass - NEEDS model in train mode
```

If `model.eval()`:
- Dropout layers are disabled → gradients may not flow correctly
- Some SNN layers may freeze → zero gradients
- Result: `grad = 0` → no perturbation!

If `model.train()`:
- All layers active → gradients flow properly
- You get meaningful gradients → actual perturbations!

**You're correct about Step 2:**
After perturbations are created, inference on adversarial images is just normal forward pass (no gradients needed there).

**Summary:**
- **Step 1 (generation)**: Model must be in train mode to compute input gradients
- **Step 2 (inference)**: Model mode doesn't matter (using Lava network anyway)

The model weights never change - train mode just enables gradient computation!

---

## Q3: How do I verify that perturbations were actually applied to the images?

**Problem:** 
The original images are 456×255 pixels but get resized to 33×100 for the model. Visual artifacts could be from compression or actual attack perturbations.

**Solution:**
The `generate_adversarial_images.py` script now creates verification images that show:

1. **Original image** (after resize to 33×100)
2. **Adversarial image** (after attack)
3. **Perturbation visualization** (normalized and color-mapped)

**What to look for:**
- Check `<ATTACK>/verification_<folder_name>/` directory
- Look at the perturbation panel (rightmost image)
- The title shows `max=X.XXXX` - this should be > 0
- Heatmap shows where perturbations are strongest (black→red→yellow→white)

**If perturbation is all black with max=0.0000:**
- The attack failed to generate perturbations
- Check that model is in `train()` mode, not `eval()` mode
- Verify gradients are being computed (see debug output)

**Perturbation Statistics:**
The script outputs:
```
Perturbation Statistics:
  Mean absolute perturbation: 0.XXXXXX
  Max absolute perturbation:  0.XXXXXX
  Epsilon (bound):            0.XX
```

The max should be close to (but not exceeding) epsilon.

---

## Q4: Why is the perturbation visualization normalized instead of just amplified?

**Normalization vs Amplification:**

**Amplification (old approach):**
```python
perturbation_vis = perturbation * 10  # Multiply by 10
```
- Problem: If perturbations are [0, 0.03], amplifying by 10 gives [0, 0.3]
- Still very dark (30% brightness)
- Hard to see spatial distribution

**Normalization (new approach):**
```python
p_min = perturbation.min()
p_max = perturbation.max()
perturbation_vis = (perturbation - p_min) / (p_max - p_min)
```
- Stretches values to use full [0, 1] range
- Smallest perturbation → black (0)
- Largest perturbation → white (1.0)
- Shows spatial distribution clearly

**Analogy:** 
It's like auto-adjusting brightness on a photo. If your photo is too dark (all pixels 0-30% brightness), normalizing stretches it to use 0-100% brightness so you can see the details.

**Important:**
The title shows the actual `max=X.XXXX` value, so you know the real magnitude while still being able to see the spatial pattern.

---

## Q5: What are the temp_data folders and why do they exist?

**Location:** `<ATTACK>/temp_data/driving_dataset/`

**Purpose:**
This is a **temporary structure** created by `run_inference_on_adversarial.py` to feed adversarial images into the Lava SDNN inference pipeline.

**Contents:**
- **Images:** Copies of adversarial images from `<ATTACK>/adv_img_<parameters>/`
- **data.txt:** Maps each image filename to its ground truth steering angle

**Why it's needed:**
The `PilotNetDataset` class expects a specific directory structure:
```
path/driving_dataset/
  ├── data.txt
  └── *.jpg files
```

Since adversarial images are stored in `<ATTACK>/adv_img_<parameters>/` but need this structure, the script:
1. Creates `temp_data/driving_dataset/` 
2. Copies adversarial images there
3. Creates `data.txt` with ground truth values
4. Runs Lava inference
5. **Deletes the entire `temp_data/` folder** after completion

**Current state:**
If you see this folder, it means either:
- The script is currently running, OR
- The script failed before reaching cleanup (line 230)

This folder is temporary and gets automatically deleted after successful inference.

---

## Q6: How does the parameter-based folder naming work?

**Folder Naming Convention:**

The adversarial image folders are automatically named based on attack parameters:

- **FGSM:** `adv_img_eps0.03` (includes epsilon only)
- **PGD:** `adv_img_eps0.03_alpha0.007_steps10` (includes epsilon, alpha, steps)
- **MIFGSM:** `adv_img_eps0.03_alpha0.007_steps10_decay1.0` (includes all parameters)

**Usage:**

**Step 1 - Generate:**
```bash
python generate_adversarial_images.py --attack FGSM --eps 0.03
```
Creates: `FGSM/adv_img_eps0.03/`

**Step 2 - Run Inference:**
```bash
python run_inference_on_adversarial.py --attack FGSM --folder adv_img_eps0.03
```

**Output Files:**
- `FGSM/results_adv_img_eps0.03.txt`
- `FGSM/comparison_adv_img_eps0.03.png`
- `FGSM/verification_adv_img_eps0.03/` (verification images)

**Benefits:**
- Easy to track which parameters were used
- Can generate multiple attacks with different parameters
- Results files clearly labeled with parameters used

---

## Q7: Why do we use `network.pt` for attacks but `network.net` for inference?

**Short Answer:**
- `network.pt` = PyTorch weights → Used for attack generation (needs gradients)
- `network.net` = Lava HDF5 model → Used for inference (no gradients needed)

**Detailed Explanation:**

Both files represent the **same trained model**, just in different formats for different purposes.

**Attack Generation uses `network.pt` (PyTorch format):**
- Need to compute **gradients w.r.t. input images** for adversarial perturbations
- Requires PyTorch's autograd system for backpropagation
- The Lava `.net` format doesn't support gradient computation
- Must use the original PyTorch model with full computational graph

**Inference uses `network.net` (Lava HDF5 format):**
- Only need **forward pass** to get predictions (no gradients)
- Optimized Lava SDNN format for efficient inference
- Can run on Loihi 2 neuromorphic hardware or CPU simulation
- More efficient for inference-only tasks

**Pipeline Overview:**

1. **Step 1 - Generate Adversarial Images:**
   - Script: `generate_adversarial_images.py`
   - Model: `Trained/network.pt` (PyTorch)
   - Purpose: Compute input gradients → create perturbations
   - Requires: Backpropagation through the model

2. **Step 2 - Run Inference on Adversarial Images:**
   - Script: `run_inference_on_adversarial.py`
   - Model: `pilotnet_sdnn/network.net` (Lava)
   - Purpose: Forward pass only → get predictions
   - Requires: No gradients, just inference

**Key Insight:**
You cannot use `network.net` for attacks because the Lava format is designed for efficient inference, not gradient computation. Similarly, while you *could* use `network.pt` for inference, `network.net` is more efficient and allows running on Loihi 2 hardware.
