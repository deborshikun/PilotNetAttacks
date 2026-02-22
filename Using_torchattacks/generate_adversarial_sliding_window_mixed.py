"""
Generate Adversarial Images using Sliding Window with Mixed Attacks

This script extends the sliding window approach by using DIFFERENT attacks for different windows.
Instead of using the same attack (e.g., PGD) for all 196 windows, it cycles through multiple
attacks that you specify.

Example with 3 attacks [FGSM, PGD, MIFGSM] and 196 windows:
- Window 0: FGSM
- Window 1: PGD
- Window 2: MIFGSM
- Window 3: FGSM (cycle repeats)
- Window 4: PGD
- ...

This tests robustness against diverse adversarial perturbations!
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


# List of available attacks in torchattacks
AVAILABLE_ATTACKS = [
    'FGSM', 'BIM', 'RFGSM', 'CW', 'PGD', 'EOTPGD', 'FFGSM', 'TPGD',
    'MIFGSM', 'UPGD', 'APGD', 'APGDT', 'FAB', 'Square', 'AutoAttack',
    'PGDL2', 'DeepFool', 'OnePixel', 'SparseFool', 'Pixle', 'GN'
]

# Default recommended attacks for regression tasks
DEFAULT_ATTACKS = ['FGSM', 'PGD', 'MIFGSM']


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
    """Wrapper to adapt torchattacks for regression tasks"""
    
    def __init__(self, attack, model, target_value):
        self.attack = attack
        self.model = model
        self.target_value = target_value
        self.device = next(model.parameters()).device
        
        # Extract attack parameters
        self.attack_name = self.attack.attack
        self.eps = self.attack.eps if hasattr(self.attack, 'eps') else 0.03
        self.steps = getattr(self.attack, 'steps', 1)
        self.alpha = getattr(self.attack, 'alpha', self.eps)
        self.random_start = getattr(self.attack, 'random_start', False)
        self.decay = getattr(self.attack, 'decay', None)
        
    def __call__(self, images):
        """Generate adversarial images for regression task"""
        images = images.clone().detach().to(self.device)
        target = torch.tensor([self.target_value]).float().to(self.device)
        
        adv_images = images.clone().detach()
        
        # Random start (PGD-style attacks)
        if self.random_start:
            delta = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images + delta, min=-1, max=1).detach()
        
        # Initialize momentum (MIFGSM-style attacks)
        momentum = torch.zeros_like(images).to(self.device) if self.decay is not None else None
        
        # Iterative attack loop
        for step in range(self.steps):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs, _, _ = self.model(adv_images)
            final_prediction = outputs.mean()
            
            # MSE loss - NEGATE to maximize distance (untargeted attack)
            cost = -nn.MSELoss()(final_prediction, target.squeeze())
            
            # Backward pass
            grad = torch.autograd.grad(cost, adv_images, 
                                     retain_graph=False, create_graph=False)[0]
            
            # Apply momentum if applicable
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


def create_attack(attack_name, model, attack_params):
    """
    Create a torchattacks attack object based on name and parameters.
    
    Args:
        attack_name: Name of the attack (e.g., 'FGSM', 'PGD', 'MIFGSM')
        model: SDNN model
        attack_params: Dictionary with attack parameters
    
    Returns:
        torchattacks attack object
    """
    eps = attack_params['eps']
    alpha = attack_params['alpha']
    steps = attack_params['steps']
    decay = attack_params['decay']
    random_start = attack_params.get('random_start', False)
    
    try:
        if attack_name == 'FGSM':
            return torchattacks.FGSM(model, eps=eps)
        elif attack_name == 'PGD':
            return torchattacks.PGD(model, eps=eps, alpha=alpha, 
                                   steps=steps, random_start=random_start)
        elif attack_name == 'MIFGSM':
            return torchattacks.MIFGSM(model, eps=eps, alpha=alpha, 
                                      steps=steps, decay=decay)
        elif attack_name == 'BIM':
            return torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'RFGSM':
            return torchattacks.RFGSM(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'TPGD':
            return torchattacks.TPGD(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'FFGSM':
            return torchattacks.FFGSM(model, eps=eps, alpha=alpha)
        elif attack_name == 'UPGD':
            return torchattacks.UPGD(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'APGD':
            return torchattacks.APGD(model, eps=eps, steps=steps)
        elif attack_name == 'APGDT':
            return torchattacks.APGDT(model, eps=eps, steps=steps)
        elif attack_name == 'EOTPGD':
            return torchattacks.EOTPGD(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'PGDL2':
            return torchattacks.PGDL2(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_name == 'DeepFool':
            return torchattacks.DeepFool(model, steps=steps)
        elif attack_name == 'CW':
            return torchattacks.CW(model, steps=steps)
        else:
            print(f"Warning: Attack '{attack_name}' not explicitly supported, falling back to FGSM")
            return torchattacks.FGSM(model, eps=eps)
    except Exception as e:
        print(f"Error creating attack '{attack_name}': {e}")
        print(f"Falling back to FGSM")
        return torchattacks.FGSM(model, eps=eps)


def generate_adversarial_sliding_window_mixed(model, original_images, attack_params, 
                                              attack_names, window_size=5):
    """
    Generate adversarial examples using sliding window with mixed attacks.
    
    Args:
        model: SDNN model
        original_images: Original image tensor [1, C, H, W, T]
        attack_params: Dictionary with attack configuration
        attack_names: List of attack names to cycle through
        window_size: Size of temporal window (default: 5)
    
    Returns:
        adversarial_images: Perturbed images [1, C, H, W, T]
        window_stats: Statistics for each window
    """
    device = next(model.parameters()).device
    T = original_images.shape[-1]
    num_windows = T - window_size + 1
    
    print(f"\n{'='*70}")
    print(f"SLIDING WINDOW ADVERSARIAL ATTACK - MIXED ATTACKS")
    print(f"{'='*70}")
    print(f"Total frames: {T}")
    print(f"Window size: {window_size}")
    print(f"Number of windows: {num_windows}")
    print(f"Attacks to use (cycling): {', '.join(attack_names)}")
    print(f"{'='*70}\n")
    
    # Initialize
    accumulated_perturbations = torch.zeros_like(original_images).to(device)
    perturbation_counts = torch.zeros(T).to(device)
    
    window_stats = []
    
    # Process each window
    for win_idx in range(num_windows):
        # Select attack for this window (cycle through the list)
        attack_name = attack_names[win_idx % len(attack_names)]
        
        print(f"Window {win_idx + 1}/{num_windows} [{attack_name}]: frames [{win_idx} to {win_idx + window_size - 1}]", end='')
        
        # Extract window
        window_start = win_idx
        window_end = win_idx + window_size
        window_frames = original_images[:, :, :, :, window_start:window_end].clone().to(device)
        
        # Get target prediction
        model.eval()
        with torch.no_grad():
            output, _, _ = model(window_frames)
            target_value = output.mean().item()
        model.train()
        
        # Create attack for this window
        attack = create_attack(attack_name, model, attack_params)
        wrapped_attack = RegressionAttackWrapper(attack, model, target_value)
        
        # Generate adversarial version
        adv_window = wrapped_attack(window_frames)
        
        # Calculate perturbation
        window_perturbation = adv_window - window_frames
        
        # Accumulate perturbation
        accumulated_perturbations[:, :, :, :, window_start:window_end] += window_perturbation
        perturbation_counts[window_start:window_end] += 1
        
        # Store statistics
        pert_norm = torch.norm(window_perturbation).item()
        window_stats.append({
            'window_idx': win_idx,
            'attack_name': attack_name,
            'frames': (window_start, window_end - 1),
            'target': target_value,
            'perturbation_norm': pert_norm
        })
        
        print(f" | Target: {target_value:.6f} | Pert: {pert_norm:.6f}")
    
    # Average perturbations
    print(f"\n{'='*70}")
    print("Averaging overlapping perturbations...")
    print(f"{'='*70}\n")
    
    perturbation_counts = torch.clamp(perturbation_counts, min=1.0)
    
    for t in range(T):
        accumulated_perturbations[:, :, :, :, t] /= perturbation_counts[t]
    
    # Apply averaged perturbations
    adversarial_images = original_images + accumulated_perturbations
    adversarial_images = torch.clamp(adversarial_images, min=-1, max=1)
    
    # Print attack distribution
    attack_counts = {}
    for stat in window_stats:
        attack_name = stat['attack_name']
        attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1
    
    print(f"Attack distribution:")
    for attack_name, count in sorted(attack_counts.items()):
        print(f"  {attack_name}: {count} windows")
    print()
    
    return adversarial_images, window_stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate adversarial images using sliding window with mixed attacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available attacks in torchattacks:
{chr(10).join('  - ' + attack for attack in AVAILABLE_ATTACKS)}

Recommended for regression tasks: {', '.join(DEFAULT_ATTACKS)}

Examples:
  # Use default attacks (FGSM, PGD, MIFGSM)
  python {os.path.basename(__file__)} --eps 0.03 --alpha 0.007 --steps 10
  
  # Use specific attacks
  python {os.path.basename(__file__)} --attacks FGSM PGD BIM --eps 0.03 --alpha 0.007 --steps 10
  
  # Use only FGSM and MIFGSM
  python {os.path.basename(__file__)} --attacks FGSM MIFGSM --eps 0.03 --alpha 0.007 --steps 10 --decay 1.0
        """
    )
    
    parser.add_argument('--attacks', type=str, nargs='+', default=DEFAULT_ATTACKS,
                        help=f'List of attacks to cycle through (default: {" ".join(DEFAULT_ATTACKS)})')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Maximum perturbation (default: 0.03)')
    parser.add_argument('--alpha', type=float, default=0.007,
                        help='Step size for iterative attacks (default: 0.007)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for iterative attacks (default: 10)')
    parser.add_argument('--decay', type=float, default=1.0,
                        help='Decay factor for momentum (MIFGSM, default: 1.0)')
    parser.add_argument('--random_start', action='store_true',
                        help='Use random start for PGD-style attacks')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Temporal window size (default: 5 frames)')
    parser.add_argument('--list-attacks', action='store_true',
                        help='List all available attacks and exit')
    
    args = parser.parse_args()
    
    # Handle --list-attacks
    if args.list_attacks:
        print("\n" + "="*70)
        print("AVAILABLE ATTACKS IN TORCHATTACKS")
        print("="*70)
        for i, attack in enumerate(AVAILABLE_ATTACKS, 1):
            print(f"  {i:2d}. {attack}")
        print("\nRecommended for regression: " + ", ".join(DEFAULT_ATTACKS))
        print("="*70 + "\n")
        return
    
    # Validate attacks
    invalid_attacks = [a for a in args.attacks if a not in AVAILABLE_ATTACKS]
    if invalid_attacks:
        print(f"\nWarning: Unknown attacks: {', '.join(invalid_attacks)}")
        print(f"Will attempt to use them anyway (may fall back to FGSM)\n")
    
    print(f"\n{'='*70}")
    print(f"SLIDING WINDOW MIXED ADVERSARIAL ATTACK")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    # Create output directory
    output_base = os.path.join(os.path.dirname(__file__), 'MIXED')
    os.makedirs(output_base, exist_ok=True)
    
    # Create folder name
    eps_str = f"{args.eps}".rstrip('0').rstrip('.').replace('.', '')
    alpha_str = f"{args.alpha}".rstrip('0').rstrip('.').replace('.', '')
    attacks_str = '_'.join(args.attacks)
    
    folder_name = f"adv_img_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_win{args.window_size}_{attacks_str}"
    
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
    
    # Stack into sequence tensor
    sequence_tensor = torch.stack(tensor_list, dim=3).unsqueeze(0).to(device)
    print(f"Sequence tensor shape: {sequence_tensor.shape}\n")
    
    # Prepare attack parameters
    attack_params = {
        'eps': args.eps,
        'alpha': args.alpha,
        'steps': args.steps,
        'decay': args.decay,
        'random_start': args.random_start
    }
    
    # Generate adversarial images using mixed sliding window
    adv_sequence, window_stats = generate_adversarial_sliding_window_mixed(
        model, sequence_tensor, attack_params, args.attacks, window_size=args.window_size
    )
    
    # Perturbation statistics
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
        img_tensor = adv_img * 0.5 + 0.5
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, img_file))
    
    print(f"Saved {len(image_files)} adversarial images\n")
    
    # Create verification images
    print("Creating verification images")
    verification_indices = (list(range(10)) + 
                          list(range(95, 105)) + 
                          list(range(len(image_files) - 10, len(image_files))))
    
    for idx in verification_indices:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        orig_img = original_images_list[idx] * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0].imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Adversarial
        adv_img = adv_sequence[0, :, :, :, idx] * 0.5 + 0.5
        adv_img = torch.clamp(adv_img, 0, 1)
        axes[1].imshow(adv_img.permute(1, 2, 0).cpu().detach().numpy())
        axes[1].set_title('Adversarial (Mixed)')
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
    stats_file = os.path.join(output_dir, 'mixed_attack_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Mixed Sliding Window Attack Statistics\n")
        f.write(f"{'='*80}\n")
        f.write(f"Attacks used: {', '.join(args.attacks)}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Epsilon: {args.eps}\n")
        f.write(f"Number of windows: {len(window_stats)}\n\n")
        
        # Attack distribution
        attack_counts = {}
        for stat in window_stats:
            attack_name = stat['attack_name']
            attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1
        
        f.write(f"Attack Distribution:\n")
        for attack_name, count in sorted(attack_counts.items()):
            f.write(f"  {attack_name}: {count} windows\n")
        f.write(f"\n")
        
        f.write(f"Window | Attack    | Frames      | Target Value | Perturbation Norm\n")
        f.write(f"{'-'*80}\n")
        for stat in window_stats:
            f.write(f"{stat['window_idx']:6d} | ")
            f.write(f"{stat['attack_name']:9s} | ")
            f.write(f"[{stat['frames'][0]:3d}, {stat['frames'][1]:3d}] | ")
            f.write(f"{stat['target']:12.6f} | ")
            f.write(f"{stat['perturbation_norm']:17.6f}\n")
    
    print(f"Statistics saved to {stats_file}")
    
    print(f"\n{'='*70}")
    print(f"Adversarial images saved to: {output_dir}/")
    print(f"Verification images: {verification_dir}/")
    print(f"\nNext step: Run inference using:")
    print(f"  python Using_torchattacks/run_inference_torchattacks.py \\")
    print(f"    --attack MIXED --folder {folder_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
