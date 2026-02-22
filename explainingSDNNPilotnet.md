# Complete Guide to SDNN PilotNet: From Training to Inference

This document provides a comprehensive, in-depth explanation of the SDNN (Sigma-Delta Neural Network) PilotNet implementation, covering every detail from training ([train.ipynb](pilotnet_sdnn/train.ipynb)) to inference ([run.ipynb](pilotnet_sdnn/run.ipynb)).

---

## Table of Contents

1. [What is SDNN?](#what-is-sdnn)
2. [PilotNet Task Overview](#pilotnet-task-overview)
3. [Dataset Structure and Preprocessing](#dataset-structure-and-preprocessing)
4. [SDNN Architecture Deep Dive](#sdnn-architecture-deep-dive)
5. [Training Process (train.ipynb)](#training-process-trainipynb)
6. [Model Export and Conversion](#model-export-and-conversion)
7. [Inference Process (run.ipynb)](#inference-process-runipynb)
8. [Temporal Processing and State Evolution](#temporal-processing-and-state-evolution)
9. [Key Differences: Training vs Inference](#key-differences-training-vs-inference)

---

## What is SDNN?

### Fundamental Concept

**Sigma-Delta Neural Networks (SDNNs)** are a class of neural networks inspired by spike-based neuromorphic computing, designed to efficiently process temporal sequences by exploiting temporal redundancy.

### Core Components

#### 1. Delta Encoder (Axon)

**Purpose:** Compute differences between consecutive activations to create sparse event-based representations.

**Mathematical Operation:**
```
delta[t] = activation[t] - activation[t-1]
```

**Thresholding:**
```
if |delta[t]| > threshold:
    send delta[t] to next layer
else:
    send nothing (no event)
```

**Key Properties:**
- **Temporal sparsity:** If input doesn't change much, delta ≈ 0 → no communication
- **Graded events:** Unlike binary spikes, delta values encode magnitude of change
- **Zero latency:** One event per timestep (unlike rate-coded SNNs requiring multiple spikes)

#### 2. Sigma Decoder (Dendrite)

**Purpose:** Accumulate incoming delta events to reconstruct the original activation values.

**Mathematical Operation:**
```
sigma[t] = sigma[t-1] + delta[t]
```

**Initialization:**
```
sigma[0] = 0  (or initial baseline)
```

**Key Properties:**
- **Lossless reconstruction:** Perfectly recovers original signal through accumulation
- **Stateful:** Maintains accumulated value across timesteps
- **Memory efficient:** Only needs to store current accumulated value

#### 3. Sigma-Delta Neuron

A complete sigma-delta neuron combines both components around a standard activation function:

```
Input → Sigma Decoder → Activation (ReLU) → Delta Encoder → Output
```

**Data Flow:**
```
Layer N-1:
    delta[t] (from previous layer)
        ↓
Sigma Decoder:
    sigma[t] = sigma[t-1] + delta[t]
        ↓
Activation:
    activated[t] = ReLU(sigma[t])
        ↓
Delta Encoder:
    delta_out[t] = activated[t] - activated[t-1]
    if |delta_out[t]| > threshold:
        send delta_out[t] to Layer N+1
```

### Why SDNN for Temporal Sequences?

**Problem with standard ANNs:**
- Process each frame independently with full computation
- Ignore temporal redundancy in video (consecutive frames are similar)
- High computational cost for every frame

**SDNN Solution:**
- **Delta encoding** exploits temporal redundancy: only encode changes
- **Event sparsity:** If scene is static, no computation in intermediate layers
- **Energy efficiency:** Fewer operations → less power consumption
- **Neuromorphic compatibility:** Can run on event-based hardware (Loihi 2)

**Example:**
```
Video frames: [Frame0, Frame1, Frame2, ...]
Changes:      [Large, Small, Tiny, ...]

Standard ANN:
  Frame0 → Full computation (100% operations)
  Frame1 → Full computation (100% operations)
  Frame2 → Full computation (100% operations)

SDNN:
  Frame0 → Full computation (100% operations, baseline)
  Frame1 → Sparse computation (20% operations, small changes)
  Frame2 → Sparse computation (5% operations, tiny changes)
```

---

## PilotNet Task Overview

### Objective

**Goal:** Predict the steering wheel angle of a car from dashboard camera images.

**Input:** RGB image from dashboard camera (456×255 pixels, resized to 100×33)

**Output:** Steering angle in radians (continuous value)

**Task Type:** Regression (not classification)

### Dataset Information

**Source:** PilotNet dataset (MIT License)
- Available at: https://github.com/lhzlhz/PilotNet/tree/master/data/datasets

**Structure:**
```
data/
├── driving_dataset/
│   ├── data.txt          # Image filenames and steering angles
│   ├── 10550.jpg         # Dashboard camera frame
│   ├── 10551.jpg
│   ├── 10552.jpg
│   └── ...
```

**data.txt Format:**
```
10550.jpg 0.0
10551.jpg 0.00872665
10552.jpg 0.0174533
...
```

Each line: `<filename> <steering_angle_in_degrees>`

**Dataset Split:**
- **Training set:** ~45,000 images
- **Testing set:** ~5,000 images

### Why Temporal Processing Makes Sense

Driving is inherently a **temporal task**:
- **Smooth transitions:** Steering angles change gradually, not abruptly
- **Temporal context:** Future steering depends on previous trajectory
- **Redundancy:** Consecutive video frames are highly similar (road, car position, etc.)

SDNN exploits this temporal structure for efficiency!

---

## Dataset Structure and Preprocessing

### PilotNetDataset Class

The custom dataset class (`pilotnet_dataset.py`) loads and preprocesses driving images.

**Initialization:**
```python
dataset = PilotNetDataset(
    path='../data',                    # Directory containing driving_dataset/
    size=(33, 100),                    # Resize target (height, width)
    transform=transforms.Compose([...]),  # Preprocessing pipeline
    train=True,                        # True for training, False for testing
    visualize=False,                   # True for sequential, False for random
    sample_offset=0,                   # Start index for sequential loading
)
```

**Key Methods:**

#### `__init__(self, path, size, transform, train, visualize, sample_offset)`

**Purpose:** Initialize dataset with parameters

**Operations:**
1. Load `data.txt` file containing filenames and steering angles
2. Parse each line: split filename and angle
3. Convert steering angles from degrees to radians: `angle_rad = angle_deg × π / 180`
4. Split into train/test based on `train` parameter
5. Store file paths and angles

#### `__len__(self)`

**Purpose:** Return number of samples in dataset

**Returns:** Integer count of images

#### `__getitem__(self, idx)`

**Purpose:** Load and preprocess a single sample

**Steps:**
1. Load image from disk: `Image.open(filepath)`
2. Convert to RGB (in case grayscale)
3. Apply transforms:
   - Resize to (33, 100)
   - Convert to PyTorch tensor: shape `[C=3, H=33, W=100]`
   - Normalize to [-1, 1]: `(pixel - 0.5) / 0.5`
4. Get corresponding steering angle (in radians)
5. Return tuple: `(image_tensor, steering_angle)`

### Data Transformations

**Training/Testing Transform:**
```python
transform = transforms.Compose([
    transforms.Resize([33, 100]),              # Resize to network input size
    transforms.ToTensor(),                      # Convert PIL Image to tensor [C, H, W]
    transforms.Normalize((0.5, 0.5, 0.5),       # Normalize each channel
                        (0.5, 0.5, 0.5)),       # to range [-1, 1]
])
```

**Normalization Details:**
```
Original pixel range: [0, 255] (uint8)
After ToTensor(): [0, 1] (float32)
After Normalize(mean=0.5, std=0.5):
    normalized = (pixel - 0.5) / 0.5
    Result: [-1, 1] range
```

**Why normalize to [-1, 1]?**
- Symmetric range around zero
- Helps gradient flow during backpropagation
- Standard for many neural networks
- Compatible with SDNN delta encoding (differences centered at zero)

### DataLoader Configuration

**Training DataLoader:**
```python
train_loader = DataLoader(
    dataset=training_set,
    batch_size=8,           # Process 8 sequences per batch
    shuffle=True,           # Randomize order each epoch
    num_workers=8           # Parallel data loading (8 threads)
)
```

**Testing DataLoader:**
```python
test_loader = DataLoader(
    dataset=testing_set,
    batch_size=8,
    shuffle=True,           # Shuffle for unbiased evaluation
    num_workers=8
)
```

**Batch Shape:**
```
Input:  [batch=8, C=3, H=33, W=100, T=sequence_length]
Target: [batch=8, T=sequence_length]
```

Where `T` is the number of frames in each temporal sequence.

---

## SDNN Architecture Deep Dive

### Network Overview

The PilotNet SDNN is a **convolutional regression network** with 10 layers implementing the following pipeline:

```
Input [3×33×100] 
  → Delta Encoding
  → Conv1 (3→24 filters)
  → Conv2 (24→36 filters)
  → Conv3 (36→64 filters)
  → Conv4 (64→64 filters)
  → Flatten
  → Dense1 (2560→100)
  → Dense2 (100→50)
  → Dense3 (50→10)
  → Output (10→1) with Sigma Decoding
  → Steering Angle
```

### Complete Network Definition

From [train.ipynb](pilotnet_sdnn/train.ipynb) Cell 7:

```python
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # Sigma-delta neuron parameters
        sdnn_params = {
            'threshold': 0.1,        # Delta unit threshold
            'tau_grad': 0.5,         # Surrogate gradient relaxation
            'scale_grad': 1,         # Surrogate gradient scale
            'requires_grad': True,   # Make threshold trainable
            'shared_param': True,    # Share threshold across layer
            'activation': F.relu,    # Activation function
        }
        
        # CNN-specific parameters (add batch normalization)
        sdnn_cnn_params = {
            **sdnn_params,
            'norm': slayer.neuron.norm.MeanOnlyBatchNorm,
        }
        
        # Dense-specific parameters (add dropout)
        sdnn_dense_params = {
            **sdnn_cnn_params,
            'dropout': slayer.neuron.Dropout(p=0.2),
        }
        
        # Network layers
        self.blocks = torch.nn.ModuleList([
            # Block 0: Input delta encoding
            slayer.block.sigma_delta.Input(sdnn_params),
            
            # Block 1: Conv layer (3 → 24 channels)
            slayer.block.sigma_delta.Conv(
                sdnn_cnn_params, 
                3, 24,           # in_channels, out_channels
                3,               # kernel_size
                padding=0,       # no padding
                stride=2,        # stride 2 (downsample)
                weight_scale=2,  # weight initialization scale
                weight_norm=True # enable weight normalization
            ),
            
            # Block 2: Conv layer (24 → 36 channels)
            slayer.block.sigma_delta.Conv(
                sdnn_cnn_params, 
                24, 36, 
                3, 
                padding=0, 
                stride=2, 
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 3: Conv layer (36 → 64 channels)
            slayer.block.sigma_delta.Conv(
                sdnn_cnn_params, 
                36, 64, 
                3, 
                padding=(1, 0),  # asymmetric padding
                stride=(2, 1),   # asymmetric stride
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 4: Conv layer (64 → 64 channels)
            slayer.block.sigma_delta.Conv(
                sdnn_cnn_params, 
                64, 64, 
                3, 
                padding=0, 
                stride=1,        # no downsampling
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 5: Flatten spatial dimensions
            slayer.block.sigma_delta.Flatten(),
            
            # Block 6: Dense layer (2560 → 100)
            slayer.block.sigma_delta.Dense(
                sdnn_dense_params, 
                64*40, 100,      # 64 channels × 40 spatial = 2560
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 7: Dense layer (100 → 50)
            slayer.block.sigma_delta.Dense(
                sdnn_dense_params, 
                100, 50, 
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 8: Dense layer (50 → 10)
            slayer.block.sigma_delta.Dense(
                sdnn_dense_params, 
                50, 10, 
                weight_scale=2, 
                weight_norm=True
            ),
            
            # Block 9: Output sigma decoding (10 → 1)
            slayer.block.sigma_delta.Output(
                sdnn_dense_params, 
                10, 1,           # 1 output (steering angle)
                weight_scale=2, 
                weight_norm=True
            )
        ])
```

### Layer-by-Layer Shape Transformation

Let's trace the shape of data through the network for a single sample:

**Input:**
```
Shape: [C=3, H=33, W=100, T=sequence_length]
```

**Block 0: Input (Delta Encoding)**
```
Operation: Compute delta[t] = input[t] - input[t-1]
Output shape: [3, 33, 100, T]  (unchanged)
```

**Block 1: Conv (3→24, kernel=3, stride=2, padding=0)**
```
Input:  [3, 33, 100, T]
Sigma decode: [3, 33, 100, T]
Conv operation:
  H_out = (33 - 3 + 2×0) / 2 + 1 = 16
  W_out = (100 - 3 + 2×0) / 2 + 1 = 49
ReLU activation: [24, 16, 49, T]
Delta encode: [24, 16, 49, T]
Output: [24, 16, 49, T]
```

**Block 2: Conv (24→36, kernel=3, stride=2, padding=0)**
```
Input:  [24, 16, 49, T]
Sigma decode: [24, 16, 49, T]
Conv operation:
  H_out = (16 - 3) / 2 + 1 = 7
  W_out = (49 - 3) / 2 + 1 = 24
ReLU: [36, 7, 24, T]
Delta encode: [36, 7, 24, T]
Output: [36, 7, 24, T]
```

**Block 3: Conv (36→64, kernel=3, stride=(2,1), padding=(1,0))**
```
Input:  [36, 7, 24, T]
Sigma decode: [36, 7, 24, T]
Conv operation:
  H_out = (7 - 3 + 2×1) / 2 + 1 = 4
  W_out = (24 - 3 + 2×0) / 1 + 1 = 22
ReLU: [64, 4, 22, T]
Delta encode: [64, 4, 22, T]
Output: [64, 4, 22, T]
```

**Block 4: Conv (64→64, kernel=3, stride=1, padding=0)**
```
Input:  [64, 4, 22, T]
Sigma decode: [64, 4, 22, T]
Conv operation:
  H_out = (4 - 3 + 0) / 1 + 1 = 2
  W_out = (22 - 3 + 0) / 1 + 1 = 20
ReLU: [64, 2, 20, T]
Delta encode: [64, 2, 20, T]
Output: [64, 2, 20, T]
```

**Block 5: Flatten**
```
Input:  [64, 2, 20, T]
Flatten spatial: [64×2×20, T] = [2560, T]
Output: [2560, T]
```

**Block 6: Dense (2560→100)**
```
Input:  [2560, T]
Sigma decode: [2560, T]
Linear + ReLU: [100, T]
Delta encode: [100, T]
Dropout (p=0.2): Randomly zero 20% of values
Output: [100, T]
```

**Block 7: Dense (100→50)**
```
Input:  [100, T]
Sigma decode: [100, T]
Linear + ReLU: [50, T]
Delta encode: [50, T]
Dropout: [50, T]
Output: [50, T]
```

**Block 8: Dense (50→10)**
```
Input:  [50, T]
Sigma decode: [50, T]
Linear + ReLU: [10, T]
Delta encode: [10, T]
Dropout: [10, T]
Output: [10, T]
```

**Block 9: Output (10→1, Sigma Decoding)**
```
Input:  [10, T]
Sigma decode: [10, T]
Linear (no activation): [1, T]
Sigma decode (output layer): [1, T]
Output: [1, T]  ← Steering angle predictions for each timestep
```

**Final Shape (with batch):**
```
Input:  [batch=8, C=3, H=33, W=100, T]
Output: [batch=8, 1, T]
```

### Detailed Block Components

#### Sigma-Delta Convolutional Block

Each convolutional block contains:

1. **Sigma Decoder (Input)**
   - Accumulates incoming deltas
   - Maintains state: `sigma[t] = sigma[t-1] + delta_in[t]`

2. **Batch Normalization** (MeanOnlyBatchNorm)
   - Normalizes across batch dimension
   - Only subtracts mean (no variance scaling)
   - Quantization-friendly

3. **Convolution**
   - Standard 2D convolution on accumulated values
   - Learnable weights and biases

4. **Activation** (ReLU)
   - `output = max(0, conv_result)`

5. **Delta Encoder (Output)**
   - Computes differences: `delta_out[t] = activated[t] - activated[t-1]`
   - Applies threshold: only send if `|delta_out[t]| > threshold`

6. **Weight Normalization**
   - Normalizes weight vectors: `w_normalized = w / ||w||`
   - Improves gradient flow and convergence

#### Sigma-Delta Dense Block

Similar to Conv block, but with:

1. **Sigma Decoder**
2. **Batch Normalization**
3. **Linear Layer** (fully connected)
4. **Activation** (ReLU)
5. **Delta Encoder**
6. **Dropout** (p=0.2)
   - Randomly zeros 20% of activations during training
   - Prevents overfitting
7. **Weight Normalization**

#### Input Block

**Special first layer:**
- Takes raw normalized images `[-1, 1]`
- Computes delta encoding: `delta[t] = image[t] - image[t-1]`
- Initialization: `image[0]` treated as baseline

#### Output Block

**Special last layer:**
- Takes accumulated features `[10, T]`
- Applies linear transformation to `[1, T]`
- **Sigma decoding** to get continuous steering predictions
- No activation function (regression output)

### Parameter Details

#### Threshold (τ)

**Purpose:** Control sparsity in delta encoder

**Value:** 0.1 (trainable)

**Effect:**
- Low threshold: More events, higher accuracy, more computation
- High threshold: Fewer events, lower accuracy, less computation

**Training:** `requires_grad=True` allows network to learn optimal threshold per layer

#### Surrogate Gradient Parameters

**Problem:** Delta encoder has non-differentiable threshold operation

**Solution:** Use surrogate gradient for backpropagation

```python
'tau_grad': 0.5,      # Controls smoothness of surrogate
'scale_grad': 1,      # Scales gradient magnitude
```

**Effect:** Enables gradient flow through threshold gates during training

#### Weight Scale

**Purpose:** Initialize weights with appropriate magnitude

**Value:** `weight_scale=2`

**Method:** Weights initialized as `W ~ Uniform(-scale/sqrt(fan_in), scale/sqrt(fan_in))`

**Reason:** Helps maintain activation magnitudes through deep network

### Forward Pass Implementation

```python
def forward(self, x):
    """
    Forward pass through SDNN
    
    Args:
        x: Input tensor [batch, C, H, W, T]
    
    Returns:
        x: Output predictions [batch, 1, T]
        event_cost: Sparsity regularization loss
        count: Event counts per layer
    """
    count = []
    event_cost = 0
    
    for block in self.blocks:
        # Process through block
        x = block(x)
        
        # Track events and compute sparsity loss
        if hasattr(block, 'neuron'):
            # Add event rate loss
            event_cost += event_rate_loss(x)
            
            # Count non-zero events (skip first timestep)
            event_count = torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item()
            count.append(event_count)
    
    print(count)  # Debug: print event counts per layer
    return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)
```

**Event Rate Loss:**
```python
def event_rate_loss(x, max_rate=0.01):
    """
    Penalize high event rates to encourage sparsity
    
    Args:
        x: Layer activations
        max_rate: Target maximum event rate (1% of values)
    
    Returns:
        Loss value (0 if below max_rate, positive if above)
    """
    mean_event_rate = torch.mean(torch.abs(x))
    penalty = F.relu(mean_event_rate - max_rate)
    return F.mse_loss(penalty, torch.zeros_like(penalty))
```

**Purpose:** Encourage network to use sparse events (only send important changes)

---

## Training Process (train.ipynb)

### Training Hyperparameters

From [train.ipynb](pilotnet_sdnn/train.ipynb) Cell 8:

```python
batch = 8              # Batch size (8 sequences per update)
lr = 0.001             # Initial learning rate
lam = 0.01             # Lagrangian for event rate loss
epochs = 20            # Total training epochs
steps = [60, 120, 160] # Learning rate reduction schedule

device = torch.device('cuda')  # Use GPU if available
```

### Optimizer Configuration

```python
optimizer = torch.optim.RAdam(
    net.parameters(), 
    lr=lr,                # Initial learning rate: 0.001
    weight_decay=1e-5     # L2 regularization: 0.00001
)
```

**RAdam (Rectified Adam):**
- Variant of Adam optimizer
- Better convergence in early training
- Adaptive learning rates per parameter
- Momentum + RMS normalization

**Weight Decay:**
- L2 regularization on weights
- Prevents overfitting
- Adds `λ||W||²` penalty to loss

### Learning Rate Schedule

```python
for epoch in range(epochs):
    if epoch in steps:  # Reduce at epochs [60, 120, 160]
        for param_group in optimizer.param_groups:
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= 10/3  # Multiply by 0.3
```

**Schedule:**
- Epoch 0-59: lr = 0.001
- Epoch 60-119: lr = 0.0003
- Epoch 120-159: lr = 0.00009
- Epoch 160+: lr = 0.000027

**Purpose:** Start with large steps for fast progress, reduce for fine-tuning

### Training Assistant

```python
stats = slayer.utils.LearningStats()  # Track metrics

assistant = slayer.utils.Assistant(
    net=net,
    error=lambda output, target: F.mse_loss(
        output.flatten(), 
        target.flatten()
    ),
    optimizer=optimizer,
    stats=stats,
    count_log=True,   # Log event counts
    lam=lam           # Sparsity loss weight: 0.01
)
```

**LearningStats:**
- Tracks training/testing loss
- Tracks accuracy metrics
- Saves statistics to files
- Generates learning curves

**Assistant:**
- Handles training/testing loops
- Computes total loss = MSE + λ×event_cost
- Performs backpropagation
- Updates optimizer
- Logs statistics

### Training Loop

From [train.ipynb](pilotnet_sdnn/train.ipynb) Cell 14:

```python
for epoch in range(epochs):
    # Learning rate reduction
    if epoch in steps:
        for param_group in optimizer.param_groups:
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= 10/3
    
    # Training phase
    for i, (input, ground_truth) in enumerate(train_loader):
        assistant.train(input, ground_truth)
        print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')
    
    # Testing phase
    for i, (input, ground_truth) in enumerate(test_loader):
        assistant.test(input, ground_truth)
        print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')
    
    # Periodic output
    if epoch % 50 == 49:
        print()
    
    # Save best model
    if stats.testing.best_loss:
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    
    # Update and save statistics
    stats.update()
    stats.save(trained_folder + '/')
    
    # Monitor gradient flow
    net.grad_flow(trained_folder + '/')
    
    # Checkpoint saves
    if epoch % 10 == 0:
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, logs_folder + f'/checkpoint{epoch}.pt')
```

### Detailed Training Steps (assistant.train)

**What happens inside `assistant.train(input, ground_truth)`:**

1. **Move data to device (GPU)**
   ```python
   input = input.to(device)        # [batch, C, H, W, T]
   ground_truth = ground_truth.to(device)  # [batch, T]
   ```

2. **Zero gradients**
   ```python
   optimizer.zero_grad()
   ```

3. **Forward pass**
   ```python
   output, event_cost, count = net(input)
   # output: [batch, 1, T]
   # event_cost: scalar (sparsity penalty)
   # count: [1, num_layers] (event counts)
   ```

4. **Compute prediction loss**
   ```python
   mse_loss = F.mse_loss(output.flatten(), ground_truth.flatten())
   ```
   
   **Flatten operation:**
   ```
   output: [batch=8, 1, T] → flatten → [8×T]
   ground_truth: [batch=8, T] → flatten → [8×T]
   MSE: mean((pred - gt)²) over all 8×T values
   ```

5. **Compute total loss**
   ```python
   total_loss = mse_loss + lam * event_cost
   # total_loss = MSE + 0.01 × sparsity_penalty
   ```
   
   **Purpose:** Balance prediction accuracy with sparsity

6. **Backward pass**
   ```python
   total_loss.backward()
   ```
   
   **Computes gradients:**
   ```
   ∂(total_loss)/∂(W1), ∂(total_loss)/∂(W2), ..., ∂(total_loss)/∂(W_last)
   ```

7. **Gradient clipping** (optional, in assistant)
   ```python
   torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
   ```
   
   **Purpose:** Prevent exploding gradients

8. **Optimizer step**
   ```python
   optimizer.step()
   ```
   
   **Updates weights:**
   ```
   W_new = W_old - lr × grad + weight_decay × W_old
   ```

9. **Update statistics**
   ```python
   stats.training.num_samples += batch
   stats.training.loss_sum += mse_loss.item()
   stats.training.correct_samples += ...  # (for classification tasks)
   ```

### Testing Loop (assistant.test)

**Similar to training, but:**
- No gradient computation: `with torch.no_grad():`
- No backward pass
- No optimizer step
- Only evaluate loss for monitoring

**Purpose:** Track generalization performance on unseen data

### Model Checkpointing

**Best Model Saving:**
```python
if stats.testing.best_loss:
    torch.save(net.state_dict(), 'Trained/network.pt')
```

**Saves only when testing loss improves:**
- Prevents overfitting
- Keeps best generalizing model

**Periodic Checkpoints:**
```python
if epoch % 10 == 0:
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f'Logs/checkpoint{epoch}.pt')
```

**Purpose:** Resume training if interrupted

### Gradient Flow Monitoring

```python
def grad_flow(self, path):
    # Extract gradient norms from each layer
    grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]
    
    # Plot on log scale
    plt.figure()
    plt.semilogy(grad)
    plt.savefig(path + 'gradFlow.png')
    plt.close()
    
    return grad
```

**Purpose:** Detect vanishing/exploding gradients
- **Vanishing:** Gradients → 0 in early layers (can't learn)
- **Exploding:** Gradients → ∞ (unstable training)

**Healthy gradient flow:** All layers have similar magnitude gradients

---

## Model Export and Conversion

### Save PyTorch Model

After training completes:

```python
# Load best model
net.load_state_dict(torch.load('Trained/network.pt'))

# Export as HDF5 for Lava
net.export_hdf5('Trained/network.net')
```

### HDF5 Export Process

```python
def export_hdf5(self, filename):
    h = h5py.File(filename, 'w')
    layer = h.create_group('layer')
    
    for i, b in enumerate(self.blocks):
        b.export_hdf5(layer.create_group(f'{i}'))
```

**What gets exported:**

For each block:
1. **Weight matrices:** Conv kernels or Dense weights
2. **Bias vectors**
3. **Neuron parameters:** Threshold, tau_grad, scale_grad
4. **Layer configuration:** Input/output sizes, stride, padding
5. **Normalization stats:** Batch norm mean/variance

**HDF5 Structure:**
```
network.net
└── layer/
    ├── 0/              # Input block
    │   ├── threshold
    │   ├── tau_grad
    │   └── ...
    ├── 1/              # Conv1
    │   ├── weight      # [24, 3, 3, 3]
    │   ├── bias        # [24]
    │   ├── threshold
    │   ├── norm/
    │   │   ├── mean
    │   │   └── ...
    │   └── ...
    ├── 2/              # Conv2
    ...
    └── 9/              # Output block
```

### network.pt vs network.net

**network.pt (PyTorch format):**
- Python dictionary with state_dict
- Loadable only with PyTorch
- Used for further training or PyTorch inference
- Contains raw parameter tensors

**network.net (HDF5 format):**
- Platform-independent binary format
- Loadable by Lava/netx module
- Used for Loihi 2 deployment
- Structured hierarchically for easy parsing

---

## Inference Process (run.ipynb)

### Overview

Inference runs the **trained SDNN model** on new driving video sequences to predict steering angles in real-time.

**Key differences from training:**
1. Uses Lava processes (event-based execution)
2. Can run on Loihi 2 neuromorphic hardware
3. No gradient computation or weight updates
4. Processes streaming data sequentially

### Import Required Modules

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 2:

```python
import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from lava.lib.dl import netx
from dataset import PilotNetDataset
from utils import (
    PilotNetEncoder,      # Input delta encoding
    PilotNetDecoder,      # Output scaling
    PilotNetMonitor,      # Real-time visualization
    CustomHwRunConfig,    # Loihi 2 configuration
    CustomSimRunConfig,   # CPU simulation configuration
    get_input_transform   # Preprocessing
)
```

### Check Loihi 2 Availability

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 4:

```python
from lava.utils.system import Loihi2
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
    print("Loihi2 compiler is not available. Execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
```

**Compression types:**
- **DELTA_SPARSE_8:** For Loihi 2 (8-bit sparse delta encoding)
- **DENSE:** For CPU (full tensor communication)

### Load Network from HDF5

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 6:

```python
net = netx.hdf5.Network(
    net_config='network.net',  # HDF5 file path
    skip_layers=1              # Skip input encoding layer (handle separately)
)
print(net)
```

**What happens:**
1. Parse HDF5 file structure
2. Create Lava process for each layer
3. Connect layers sequentially
4. Load weights and parameters
5. Return ready-to-run network

**Network structure:**
```python
print(f'There are {len(net)} layers in the network:')

for l in net.layers:
    print(f'{l.__class__.__name__:5s} : {l.name:10s}, shape : {l.shape}')
```

**Output:**
```
Conv  : layer0    , shape : (24, 16, 49)
Conv  : layer1    , shape : (36, 7, 24)
Conv  : layer2    , shape : (64, 4, 22)
Conv  : layer3    , shape : (64, 2, 20)
Flatten: layer4   , shape : (2560,)
Dense : layer5    , shape : (100,)
Dense : layer6    , shape : (50,)
Dense : layer7    , shape : (10,)
Dense : layer8    , shape : (1,)
```

**Layer count:** 9 layers (skipped input layer)

### Configure Execution Parameters

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 8:

```python
num_samples = 200              # Number of frames to process
steps_per_sample = 1           # 1 timestep per frame
num_steps = num_samples + len(net.layers)  # 200 + 9 = 209
out_offset = len(net.layers) + 3           # 9 + 3 = 12
```

**Why num_steps = 209?**
- Process 200 frames
- Need extra timesteps for signal propagation through 9 layers
- Total: 200 + 9 = 209

**Why out_offset = 12?**
- Output prediction delayed by layer propagation
- First prediction appears at timestep `out_offset`
- Empirically: 9 layers + 3 buffer = 12 timestep delay

**Temporal alignment:**
```
Timestep 0:   Input frame 0    → No output yet (propagating)
Timestep 1:   Input frame 1    → No output yet
...
Timestep 12:  Input frame 12   → First output (for frame 0)
Timestep 13:  Input frame 13   → Second output (for frame 1)
...
Timestep 211: (padding)        → Last output (for frame 199)
```

### Create Dataset

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 9-10:

```python
transform = get_input_transform(net.net_config)

full_set = PilotNetDataset(
    path='../data',
    size=net.inp.shape[:2],      # (33, 100)
    transform=transform,
    visualize=True,              # Sequential loading
    sample_offset=10550,         # Start from frame 10550
)
```

**Dataset returns:**
- Images: Frames 10550, 10551, ..., 10749 (200 frames)
- Labels: Corresponding steering angles

**visualize=True:** Returns frames in order (not shuffled)

### Create Lava Processes

#### 1. Dataloader

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 11:

```python
dataloader = io.dataloader.SpikeDataloader(dataset=full_set)
```

**Purpose:** Read images from dataset and convert to spikes

**Outputs:**
- `s_out`: Spike-encoded image data
- `ground_truth`: Steering angle labels

**Process:**
```
Dataset → Load image → Convert to events → Send to network
         → Load label → Send to logger
```

#### 2. Input Encoder

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 13:

```python
input_encoder = PilotNetEncoder(
    shape=net.inp.shape,         # (3, 33, 100)
    net_config=net.net_config,   # Network configuration
    compression=compression      # DENSE or DELTA_SPARSE_8
)
```

**Purpose:** Delta encode consecutive frames

**Operation:**
```python
# Pseudocode for PilotNetEncoder
class PilotNetEncoder:
    def __init__(self, shape, net_config, compression):
        self.prev_frame = None
        self.compression = compression
    
    def encode(self, frame):
        if self.prev_frame is None:
            delta = frame  # First frame: send as-is
        else:
            delta = frame - self.prev_frame  # Delta encoding
        
        self.prev_frame = frame
        
        if self.compression == DELTA_SPARSE_8:
            return compress_sparse(delta)  # Send only non-zero deltas
        else:
            return delta  # Send full delta tensor
```

**Output:** Delta-encoded frames ready for SDNN

#### 3. Network (SDNN)

```python
net = netx.hdf5.Network(net_config='network.net', skip_layers=1)
```

**Already loaded!** See "Load Network from HDF5" section above.

**Inputs:** Delta-encoded frames

**Outputs:** Raw steering predictions

#### 4. Output Decoder

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 15:

```python
output_decoder = PilotNetDecoder(shape=net.out.shape)
```

**Purpose:** Scale and convert network output to steering angle

**Operation:**
```python
# Pseudocode for PilotNetDecoder
class PilotNetDecoder:
    def __init__(self, shape):
        self.scale = get_output_scale()  # From net_config
    
    def decode(self, raw_output):
        return raw_output * self.scale
```

**Input:** [1] raw network output

**Output:** [1] steering angle in radians

#### 5. Monitor (Visualization)

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 17:

```python
monitor = PilotNetMonitor(
    shape=net.inp.shape,
    transform=transform,
    output_offset=out_offset  # 12 timestep delay
)
```

**Purpose:** Real-time visualization during inference

**Displays:**
- Current input frame
- Ground truth steering angle
- Predicted steering angle
- Comparison plot

**Updates every timestep** with new predictions

#### 6. Data Loggers

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 17:

```python
gt_logger = io.sink.RingBuffer(
    shape=(1,),          # Single value per timestep
    buffer=num_steps     # Store 209 timesteps
)

output_logger = io.sink.RingBuffer(
    shape=net.out_layer.shape,  # (1,)
    buffer=num_steps            # Store 209 timesteps
)
```

**Purpose:** Store all predictions and ground truths for post-processing

**RingBuffer:**
- Fixed-size circular buffer
- Stores timestamped values
- Accessible after run completes

### Connect Processes

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 19:

```python
# Ground truth logging
dataloader.ground_truth.connect(gt_logger.a_in)

# Main processing pipeline
dataloader.s_out.connect(input_encoder.inp)
input_encoder.out.connect(net.inp)
net.out.connect(output_decoder.inp)
output_decoder.out.connect(output_logger.a_in)

# Monitor connections
dataloader.s_out.connect(monitor.frame_in)
dataloader.ground_truth.connect(monitor.gt_in)
output_decoder.out.connect(monitor.output_in)
```

**Data flow diagram:**
```
┌─────────────┐
│  Dataloader │
└──┬────────┬─┘
   │        │
   │        └──────────────┐
   │                       │
   ▼                       ▼
┌──────────────┐    ┌─────────────┐
│Input Encoder │    │  GT Logger  │
└──────┬───────┘    └─────────────┘
       │
       ▼
┌─────────────┐
│   Network   │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│Output Decoder│
└──────┬───────┘
       │
       ├─────────────┐
       ▼             ▼
┌──────────────┐ ┌─────────┐
│Output Logger │ │ Monitor │
└──────────────┘ └─────────┘
```

### Run Inference

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 20:

```python
# Select backend
if loihi2_is_available:
    run_config = CustomHwRunConfig()  # Loihi 2 hardware
else:
    run_config = CustomSimRunConfig()  # CPU simulation

# Run network
net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)

# Retrieve results
output = output_logger.data.get().flatten()
gts = gt_logger.data.get().flatten()

# Stop execution
net.stop()
```

**RunSteps(num_steps=209):**
- Execute network for 209 timesteps
- Process 200 frames + propagation delay

**Backend selection:**
- **Loihi 2:** Event-based neuromorphic hardware execution
- **CPU:** Standard simulation for testing

**Results:**
```
output: [209] - Steering predictions for each timestep
gts:    [209] - Ground truth for each timestep
```

### Extract Valid Predictions

```python
max_outputs = len(output) - out_offset  # 209 - 12 = 197
num_to_save = min(num_samples, max_outputs)  # min(200, 197) = 197
```

**Account for output delay:**
```python
for idx in range(num_to_save):
    img_name = image_names[idx + full_set.sample_offset]
    gt = gts[idx]
    out = output[out_offset + idx]  # Shift by 12 timesteps
```

**Mapping:**
```
Image 10550 (idx=0) → gt=gts[0], pred=output[12]
Image 10551 (idx=1) → gt=gts[1], pred=output[13]
...
Image 10746 (idx=196) → gt=gts[196], pred=output[208]
```

**Last 3 frames not predicted** due to insufficient timesteps

### Save Results

```python
with open("results.txt", "w") as f:
    f.write("ImageName\t\tGroundTruth\t\tOutput\n")
    for idx in range(num_to_save):
        img_name = image_names[idx + full_set.sample_offset]
        gt = gts[idx]
        out = output[out_offset + idx]
        f.write(f"{img_name}\t{gt}\t\t\t{out}\n")
```

**results.txt format:**
```
ImageName       GroundTruth     Output
10550.jpg       0.0             -0.000015
10551.jpg       0.00872         0.000123
10552.jpg       0.01745         0.001456
...
```

### Visualize Results

From [run.ipynb](pilotnet_sdnn/run.ipynb) Cell 22:

```python
plt.figure(figsize=(7, 5))
plt.plot(np.array(gts), label='Ground Truth')
plt.plot(np.array(output[out_offset:]).flatten(), label='Lava output')
plt.xlabel(f'Sample frames (+10550)')
plt.ylabel('Steering angle (radians)')
plt.legend()
```

**Plot shows:**
- Blue line: Ground truth steering angles
- Orange line: SDNN predictions
- X-axis: Frame number (relative to 10550)
- Y-axis: Steering angle in radians

**Good predictions:** Orange follows blue closely

---

## Temporal Processing and State Evolution

### Delta Encoding Mechanism

**Input Layer (Block 0):**

```python
class DeltaEncoder:
    def __init__(self):
        self.prev_input = None
    
    def forward(self, x):
        # x shape: [batch, C, H, W, T]
        batch, C, H, W, T = x.shape
        
        delta = torch.zeros_like(x)
        
        # First timestep: send input as-is
        delta[..., 0] = x[..., 0]
        
        # Subsequent timesteps: compute differences
        for t in range(1, T):
            delta[..., t] = x[..., t] - x[..., t-1]
        
        return delta
```

**Example:**
```
Input frames (normalized [-1, 1]):
  frame[0] = [0.5, 0.6, 0.4, ...]
  frame[1] = [0.5, 0.7, 0.4, ...]
  frame[2] = [0.6, 0.7, 0.3, ...]

Delta encoding:
  delta[0] = [0.5, 0.6, 0.4, ...]      (baseline)
  delta[1] = [0.0, 0.1, 0.0, ...]      (changes only)
  delta[2] = [0.1, 0.0, -0.1, ...]     (changes only)
```

**Sparsity benefit:**
- If frame unchanged: delta = [0, 0, 0, ...] → no processing in next layer
- If small changes: delta = [0.01, 0, -0.02, ...] → sparse processing

### Sigma Decoding Mechanism

**Conv/Dense Layers (Blocks 1-9):**

```python
class SigmaDecoder:
    def __init__(self):
        self.sigma = None
    
    def forward(self, delta):
        # delta shape: [batch, C, H, W, T]
        
        if self.sigma is None:
            # Initialize accumulator
            self.sigma = torch.zeros_like(delta[..., 0])
        
        batch, C, H, W, T = delta.shape
        reconstructed = torch.zeros_like(delta)
        
        for t in range(T):
            # Accumulate delta
            self.sigma = self.sigma + delta[..., t]
            
            # Store reconstructed value
            reconstructed[..., t] = self.sigma
        
        return reconstructed
```

**Example:**
```
Incoming deltas:
  delta[0] = [0.5, 0.6, 0.4]
  delta[1] = [0.0, 0.1, 0.0]
  delta[2] = [0.1, 0.0, -0.1]

Sigma accumulation:
  sigma[0] = 0 + delta[0] = [0.5, 0.6, 0.4]
  sigma[1] = [0.5, 0.6, 0.4] + delta[1] = [0.5, 0.7, 0.4]
  sigma[2] = [0.5, 0.7, 0.4] + delta[2] = [0.6, 0.7, 0.3]

Reconstructed values = sigma
```

**Lossless reconstruction:** Original signal perfectly recovered!

### Complete Temporal Flow Through One Block

**Block 1 (Conv layer):**

```
Timestep 0:
  delta_in[0] → sigma[0] = delta[0] → conv → relu → activated[0]
                                              → delta_out[0] = activated[0]

Timestep 1:
  delta_in[1] → sigma[1] = sigma[0] + delta[1] → conv → relu → activated[1]
                                                          → delta_out[1] = activated[1] - activated[0]

Timestep 2:
  delta_in[2] → sigma[2] = sigma[1] + delta[2] → conv → relu → activated[2]
                                                          → delta_out[2] = activated[2] - activated[1]
```

**State evolution:**
- **Sigma state** accumulates over time
- **Delta output** only sends changes
- **Sparsity** increases when activations stabilize

### Multi-Layer Temporal Dependencies

**Full network temporal flow:**

```
Frame 0 input:
  Block 0: delta_0 → sigma_0
  Block 1: delta_0 → sigma_0 → delta_0
  Block 2: delta_0 → sigma_0 → delta_0
  ...
  Block 9: δ_0 → σ_0 → prediction[0]

Frame 1 input:
  Block 0: delta_1
  Block 1: delta_1 + accumulated_state_1
  Block 2: delta_1 + accumulated_state_1
  ...
  Block 9: prediction[1] depends on all previous state

Frame t input:
  Each layer maintains state from frames [0...t-1]
  Prediction[t] depends on entire temporal history
```

**Temporal window effect:**
- Prediction at timestep t uses context from ALL previous frames [0...t]
- Early frames have limited context → weaker predictions
- Later frames have full context → stronger predictions

---

## Key Differences: Training vs Inference

### Architecture

| Aspect | Training (train.ipynb) | Inference (run.ipynb) |
|--------|------------------------|----------------------|
| **Framework** | PyTorch (native Python) | Lava (event-based execution) |
| **Network format** | Python Network class | HDF5 netx.hdf5.Network |
| **Layer implementation** | slayer.block modules | Lava processes |
| **Data representation** | Dense tensors [B,C,H,W,T] | Event streams (optional sparse) |

### Execution

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Gradients** | ✅ Computed (backprop) | ❌ Not computed |
| **Weight updates** | ✅ Via optimizer.step() | ❌ Frozen |
| **Dropout** | ✅ Active (p=0.2) | ❌ Disabled |
| **Batch norm** | ✅ Uses batch stats | ✅ Uses running stats (from training) |
| **Backend** | GPU (CUDA) | CPU simulation or Loihi 2 hardware |

### Data Processing

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input encoding** | Built into Block 0 | Separate PilotNetEncoder process |
| **Output decoding** | Built into Block 9 | Separate PilotNetDecoder process |
| **Data loading** | DataLoader (batch) | SpikeDataloader (streaming) |
| **Shuffling** | ✅ Random order | ❌ Sequential order |

### Temporal Processing

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Sequence length** | Variable (depends on dataset) | Fixed (200 frames) |
| **Timesteps** | T = sequence length | T = 200 + 9 = 209 |
| **Output delay** | None (aligned) | 12 timesteps (out_offset) |
| **State reset** | Per batch | Continuous across sequence |

### Outputs

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Predictions** | [batch, 1, T] | [1, 1, 209] |
| **Loss** | MSE + λ×event_cost | Not computed |
| **Metrics** | Training/testing loss | Comparison with ground truth |
| **Event counts** | Logged and penalized | Logged (no penalty) |

### Purpose

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Learn optimal weights | Predict steering angles |
| **Optimization** | Minimize prediction error + sparsity | N/A |
| **Hardware** | GPU for fast training | Loihi 2 for efficient inference |
| **Real-time** | ❌ Offline batch processing | ✅ Real-time streaming |

---

## Summary

### Complete Pipeline Overview

**Training Phase:**
1. Load PilotNet dataset (45k training, 5k testing images)
2. Create SDNN with 10 blocks (input → 4 conv → flatten → 3 dense → output)
3. Train for 20 epochs with RAdam optimizer
4. Use MSE loss + event sparsity regularization
5. Save best model as network.pt
6. Export to network.net (HDF5) for Lava deployment

**Inference Phase:**
1. Load network.net into Lava
2. Create processing pipeline (dataloader → encoder → network → decoder → logger)
3. Connect processes in execution graph
4. Run on CPU simulation or Loihi 2 hardware
5. Process 200 frames with temporal state evolution
6. Extract predictions accounting for 12-timestep delay
7. Evaluate and visualize results

### Key Insights

**SDNN Advantages:**
- ✅ **Temporal efficiency:** Delta encoding reduces computation by 70-90%
- ✅ **Neuromorphic compatible:** Runs on event-based hardware (Loihi 2)
- ✅ **Lossless:** Sigma decoding perfectly reconstructs signals
- ✅ **No latency:** One event per timestep (unlike rate coding)

**Temporal Processing:**
- 🕐 **Stateful:** Each layer maintains accumulated sigma values
- 🕐 **Causal:** Prediction at time t depends on all frames [0...t]
- 🕐 **Delayed:** Output appears 12 timesteps after input (propagation)
- 🕐 **Context-dependent:** More frames → better predictions

**Training vs Inference:**
- 🎓 **Training:** Update weights, compute gradients, batch processing
- 🚀 **Inference:** Frozen weights, streaming data, real-time execution
- 🔄 **Both:** Use same SDNN architecture with temporal state evolution

---

## Appendix: Mathematical Formulation

### Delta Encoding

$$\delta_t = \begin{cases} 
x_0 & \text{if } t = 0 \\
x_t - x_{t-1} & \text{if } t > 0
\end{cases}$$

### Sigma Decoding

$$\sigma_t = \sigma_{t-1} + \delta_t = \sum_{i=0}^{t} \delta_i$$

### Thresholded Delta Encoder

$$\delta_{out,t} = \begin{cases}
a_t - a_{t-1} & \text{if } |a_t - a_{t-1}| > \tau \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $a_t$ = activation at time t
- $\tau$ = threshold parameter (trainable)

### Complete Layer Operation

$$h_t = \text{ReLU}(\text{Conv}(\sigma_t))$$
$$\sigma_t = \sigma_{t-1} + \delta_{in,t}$$
$$\delta_{out,t} = h_t - h_{t-1}$$

### Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{sparsity}$$

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

$$\mathcal{L}_{sparsity} = \text{ReLU}(\text{mean}(|\delta|) - r_{max})$$

Where:
- $y_i$ = ground truth steering angle
- $\hat{y}_i$ = predicted steering angle
- $\lambda$ = 0.01 (sparsity weight)
- $r_{max}$ = 0.01 (target event rate)

### Gradient Flow (Backpropagation Through Time)

$$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W}$$

Where temporal dependencies create gradient accumulation across all timesteps.

---

**End of Document**

This comprehensive guide covers every aspect of SDNN PilotNet from training to inference. For questions or clarifications, refer to the [FAQ.md](FAQ.md) document.
