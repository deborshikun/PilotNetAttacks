"""
Generate Adversarial Images using Sliding Window Approach

This script implements the sliding window adversarial attack approach.
Instead of perturbing all 200 frames as one sequence, it treats overlapping 5-frame windows
as independent experiments, attacking each window separately.

Key Difference from generate_adversarial_torchattacks.py:
- Old: Attack entire 200-frame sequence at once
- New: Attack overlapping 5-frame windows independently (196 separate attacks)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lava.lib.dl.slayer as slayer


class Network(nn.Module):
    """PilotNet SDNN Model Architecture"""
    
    def __init__(self):
        super(Network, self).__init__()
        sdnn_params = {
            'threshold': 0.1, 'tau_grad': 0.5, 'scale_grad': 1,
            'requires_grad': True, 'shared_param': True, 'activation': F.relu,
        }
        sdnn_cnn_params = {
            **sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm,
        }
        sdnn_dense_params = {
            **sdnn_cnn_params, 'dropout': slayer.neuron.Dropout(p=0.2),
        }
        self.blocks = nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 3, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 64*40, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 100, 50, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 50, 10, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Output(sdnn_dense_params, 10, 1, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x):
        """Forward pass through SDNN"""
        for block in self.blocks:
            x = block(x)
        return x, None, None


class RegressionAttackWrapper:
    """
    Wrapper to adapt torchattacks for regression tasks.
    Same as in generate_adversarial_torchattacks.py
    """
    def __init__(self, attack, model, target_value):
        self.attack = attack
        self.model = model
        self.target_value = target_value
        self.device = next(model.parameters()).device
        
        # Extract attack parameters from torchattacks object
        self.attack_name = self.attack.attack
        self.eps = self.attack.eps
        self.steps = getattr(self.attack, 'steps', 1)
        self.alpha = getattr(self.attack, 'alpha', self.eps)
        self.random_start = getattr(self.attack, 'random_start', False)
        self.decay = getattr(self.attack, 'decay', None)
        
    def __call__(self, images):
        """Generate adversarial images for regression task"""
        images = images.clone().detach().to(self.device)
        target = torch.tensor([self.target_value]).float().to(self.device)
        
        adv_images = images.clone().detach()
        
        # Random start (PGD)
        if self.random_start:
            delta = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images + delta, min=-1, max=1).detach()
        
        # Initialize momentum (MIFGSM)
        momentum = torch.zeros_like(images).to(self.device) if self.decay is not None else None
        
        # Iterative attack loop
        for step in range(self.steps):
            adv_images.requires_grad = True
            
            # Forward pass through SDNN model
            outputs, _, _ = self.model(adv_images)
            final_prediction = outputs.mean()
            
            # MSE loss for regression (we want to MAXIMIZE this to create adversarial examples)
            # Use negative loss so gradient descent becomes gradient ascent
            cost = -nn.MSELoss()(final_prediction, target.squeeze())
            
            # Backward pass
            grad = torch.autograd.grad(cost, adv_images, 
                                     retain_graph=False, create_graph=False)[0]
            
            # Apply momentum if this is MIFGSM
            if momentum is not None:
                grad_norm = torch.norm(grad, p=1)
                grad = grad / (grad_norm + 1e-8)
                grad = grad + self.decay * momentum
                momentum = grad.clone()
            
            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
        
        return adv_images


def generate_adversarial_sliding_window(model, original_images, attack_params, window_size=5):
    """
    Generate adversarial examples using sliding window approach.

    Instead of attacking all frames at once, we:
    1. Create overlapping windows of size 'window_size' (e.g., 5 frames)
    2. Attack each window independently 
    3. Accumulate perturbations across overlapping windows
    
    Example with 200 frames and window_size=5:
    - Window 0: frames [0,1,2,3,4]    → Generate adversarial version
    - Window 1: frames [1,2,3,4,5]    → Generate adversarial version
    - Window 2: frames [2,3,4,5,6]    → Generate adversarial version
    - ...
    - Window 195: frames [195,196,197,198,199] → Generate adversarial version
    """
    device = next(model.parameters()).device
    T = original_images.shape[-1]  # Total number of frames (e.g., 200)
    num_windows = T - window_size + 1  # Number of windows (e.g., 196 for 200 frames)
    
    print(f"\n{'='*70}")
    print(f"SLIDING WINDOW ADVERSARIAL ATTACK")
    print(f"{'='*70}")
    print(f"Total frames: {T}")
    print(f"Window size: {window_size}")
    print(f"Number of windows: {num_windows}")
    print(f"{'='*70}\n")
    
    # Initialize: Start with original images
    # We'll accumulate perturbations from each window
    accumulated_perturbations = torch.zeros_like(original_images).to(device)
    perturbation_counts = torch.zeros(T).to(device)  # Track how many times each frame is perturbed
    
    window_stats = []
    
    # Process each window
    for win_idx in range(num_windows):
        print(f"Processing Window {win_idx + 1}/{num_windows}: frames [{win_idx} to {win_idx + window_size - 1}]", end='')
        
        # Extract window of frames: shape [1, C, H, W, window_size]
        window_start = win_idx
        window_end = win_idx + window_size
        window_frames = original_images[:, :, :, :, window_start:window_end].clone().to(device)
        
        # Step 1: Get target prediction for THIS window
        model.eval()
        with torch.no_grad():
            output, _, _ = model(window_frames)
            target_value = output.mean().item()
        model.train()
        
        # Step 2: Create attack for THIS window
        if attack_params['attack'] == 'FGSM':
            attack = torchattacks.FGSM(model, eps=attack_params['eps'])
        elif attack_params['attack'] == 'PGD':
            attack = torchattacks.PGD(model, eps=attack_params['eps'], 
                                     alpha=attack_params['alpha'],
                                     steps=attack_params['steps'],
                                     random_start=attack_params.get('random_start', False))
        elif attack_params['attack'] == 'MIFGSM':
            attack = torchattacks.MIFGSM(model, eps=attack_params['eps'],
                                        alpha=attack_params['alpha'],
                                        steps=attack_params['steps'],
                                        decay=attack_params['decay'])
        
        wrapped_attack = RegressionAttackWrapper(attack, model, target_value)
        
        # Step 3: Generate adversarial version of THIS window
        adv_window = wrapped_attack(window_frames)
        
        # Step 4: Calculate perturbation for THIS window
        window_perturbation = adv_window - window_frames
        
        # Step 5: Accumulate perturbation for frames in this window
        accumulated_perturbations[:, :, :, :, window_start:window_end] += window_perturbation
        perturbation_counts[window_start:window_end] += 1
        
        # Store statistics
        pert_norm = torch.norm(window_perturbation).item()
        window_stats.append({
            'window_idx': win_idx,
            'frames': (window_start, window_end - 1),
            'target': target_value,
            'perturbation_norm': pert_norm
        })
        
        print(f" | Target: {target_value:.6f} | Pert norm: {pert_norm:.6f}")
    
    # Step 6: Average perturbations across overlapping windows
    # Each frame was perturbed multiple times, so we average them
    print(f"\n{'='*70}")
    print("Averaging overlapping perturbations...")
    print(f"{'='*70}\n")
    
    # Avoid division by zero (shouldn't happen, but safety first)
    perturbation_counts = torch.clamp(perturbation_counts, min=1.0)
    
    # Average: divide accumulated perturbations by count
    # Broadcasting: [1, C, H, W, T] / [T] → need to reshape
    for t in range(T):
        accumulated_perturbations[:, :, :, :, t] /= perturbation_counts[t]
    
    # Step 7: Apply averaged perturbations to original images
    adversarial_images = original_images + accumulated_perturbations
    adversarial_images = torch.clamp(adversarial_images, min=-1, max=1)
    
    # Print statistics about how many times each frame was attacked
    print(f"Perturbation count per frame:")
    print(f"  Frames 0-{window_size-2}: attacked {[perturbation_counts[i].item() for i in range(min(window_size-1, T))]} times")
    print(f"  Frames {window_size-1}-{T-window_size}: attacked {int(perturbation_counts[window_size-1].item())} times each")
    print(f"  Frames {T-window_size+1}-{T-1}: attacked {[perturbation_counts[i].item() for i in range(max(0, T-window_size+1), T)]} times\n")
    
    return adversarial_images, window_stats


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial images using sliding window approach')
    parser.add_argument('--attack', type=str, required=True, choices=['FGSM', 'PGD', 'MIFGSM'],
                        help='Attack method')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Maximum perturbation')
    parser.add_argument('--alpha', type=float, default=0.007,
                        help='Step size for iterative attacks')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for iterative attacks')
    parser.add_argument('--decay', type=float, default=1.0,
                        help='Decay factor for momentum (MIFGSM only)')
    parser.add_argument('--random_start', action='store_true',
                        help='Use random start for PGD')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Temporal window size (default: 5 frames)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SLIDING WINDOW ADVERSARIAL ATTACK - {args.attack}")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    # Create output directory
    output_base = os.path.join(os.path.dirname(__file__), args.attack)
    os.makedirs(output_base, exist_ok=True)
    
    # Create folder name with window size indicator
    eps_str = f"{args.eps}".rstrip('0').rstrip('.').replace('.', '')
    alpha_str = f"{args.alpha}".rstrip('0').rstrip('.').replace('.', '')
    decay_str = f"{args.decay}".rstrip('0').rstrip('.').replace('.', '')
    
    if args.attack == 'FGSM':
        folder_name = f"adv_img_eps{eps_str}_win{args.window_size}"
    elif args.attack == 'PGD':
        folder_name = f"adv_img_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_win{args.window_size}"
        if args.random_start:
            folder_name += "_randstart"
    elif args.attack == 'MIFGSM':
        folder_name = f"adv_img_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_decay{decay_str}_win{args.window_size}"
    
    output_dir = os.path.join(output_base, folder_name)
    verification_dir = os.path.join(output_base, f'verification_{folder_name}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
    # Load model
    print("Loading model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = Network().to(device)
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.train()
    print("Model loaded\n")
    
    # Load and preprocess images
    print("Loading images from testing_dataset")
    image_files = sorted([f for f in os.listdir(testing_dataset_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('.')[0]))
    
    transform_to_tensor = transforms.Compose([
        transforms.Resize((33, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    tensor_list = []
    original_images_list = []
    
    for img_file in image_files:
        img_path = os.path.join(testing_dataset_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform_to_tensor(img)
        tensor_list.append(img_tensor)
        original_images_list.append(img_tensor.clone())
    
    if len(tensor_list) == 0:
        print("No images found in testing_dataset")
        return
    
    print(f"Loaded {len(tensor_list)} images\n")
    
    # Stack into sequence tensor [1, C, H, W, T]
    sequence_tensor = torch.stack(tensor_list, dim=3).unsqueeze(0).to(device)
    print(f"Sequence tensor shape: {sequence_tensor.shape}\n")
    
    # Prepare attack parameters
    attack_params = {
        'attack': args.attack,
        'eps': args.eps,
        'alpha': args.alpha,
        'steps': args.steps,
        'decay': args.decay,
        'random_start': args.random_start
    }
    
    # Generate adversarial images using sliding window approach
    adv_sequence, window_stats = generate_adversarial_sliding_window(
        model, sequence_tensor, attack_params, window_size=args.window_size
    )
    
    # Calculate overall perturbation statistics
    perturbation = torch.abs(adv_sequence - sequence_tensor)
    print(f"{'='*70}")
    print("Overall Perturbation Statistics:")
    print(f"{'='*70}")
    print(f"Mean absolute perturbation: {perturbation.mean():.6f}")
    print(f"Max absolute perturbation:  {perturbation.max():.6f}")
    print(f"Epsilon (bound):            {args.eps}\n")
    
    # Save adversarial images
    print(f"Saving adversarial images to {output_dir}/")
    for i, img_file in enumerate(image_files):
        adv_img = adv_sequence[0, :, :, :, i]
        img_tensor = adv_img * 0.5 + 0.5  # Denormalize
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, img_file))
    
    print(f"Saved {len(image_files)} adversarial images\n")
    
    # Create verification images for first 10, middle 10, and last 10 frames
    print("Creating verification images")
    verification_indices = (list(range(10)) +  # First 10
                          list(range(95, 105)) +  # Middle 10
                          list(range(len(image_files) - 10, len(image_files))))  # Last 10
    
    for idx in verification_indices:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        orig_img = original_images_list[idx] * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0].imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Adversarial image
        adv_img = adv_sequence[0, :, :, :, idx] * 0.5 + 0.5
        adv_img = torch.clamp(adv_img, 0, 1)
        axes[1].imshow(adv_img.permute(1, 2, 0).cpu().detach().numpy())
        axes[1].set_title('Adversarial')
        axes[1].axis('off')
        
        # Perturbation
        pert = torch.abs(adv_sequence[0, :, :, :, idx] - original_images_list[idx].to(device))
        p_min, p_max = pert.min(), pert.max()
        if p_max > p_min:
            pert_vis = (pert - p_min) / (p_max - p_min)
        else:
            pert_vis = pert
        
        axes[2].imshow(pert_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[2].set_title(f'Perturbation (max={p_max:.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(verification_dir, f'verification_{idx:03d}_{image_files[idx]}'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Created {len(verification_indices)} verification images\n")
    
    # Save window statistics
    stats_file = os.path.join(output_dir, 'window_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Sliding Window Attack Statistics\n")
        f.write(f"{'='*70}\n")
        f.write(f"Attack: {args.attack}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Epsilon: {args.eps}\n")
        f.write(f"Number of windows: {len(window_stats)}\n\n")
        f.write(f"Window | Frames      | Target Value | Perturbation Norm\n")
        f.write(f"{'-'*70}\n")
        for stat in window_stats:
            f.write(f"{stat['window_idx']:6d} | ")
            f.write(f"[{stat['frames'][0]:3d}, {stat['frames'][1]:3d}] | ")
            f.write(f"{stat['target']:12.6f} | ")
            f.write(f"{stat['perturbation_norm']:17.6f}\n")
    
    print(f"Window statistics saved to {stats_file}")
    
    print(f"\n{'='*70}")
    print(f"Adversarial images saved to: {output_dir}/")
    print(f"Verification images: {verification_dir}/")
    print(f"\nNext step: Run inference using:")
    print(f"  python Using_torchattacks/run_inference_torchattacks.py \\")
    print(f"    --attack {args.attack} --folder {folder_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
