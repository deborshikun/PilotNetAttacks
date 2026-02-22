"""
Single Frame Perturbation Analysis - Growing Window Approach

For each iteration i (1 to 200):
1. Run PyTorch inference on clean frames [0 to i]
2. Run PyTorch inference on frames [0 to i-1 clean, i perturbed]
3. Compare the two steering angles

This reveals the incremental impact of each frame as it's added to the sequence.
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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class Network(nn.Module):
    """PilotNet SDNN Model Architecture"""
    
    def __init__(self):
        super(Network, self).__init__()
        # Import slayer locally to avoid dependency issues
        import lava.lib.dl.slayer as slayer
        
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
            
            # Forward pass
            outputs, _, _ = self.model(adv_images)
            # SDNN Output: [batch, features, time] - use LAST timestep
            final_prediction = outputs[0, 0, -1]
            
            # MSE loss - negated for untargeted attack
            cost = -nn.MSELoss()(final_prediction, target.squeeze())
            
            # Backward pass
            grad = torch.autograd.grad(cost, adv_images, 
                                     retain_graph=False, create_graph=False)[0]
            
            # Apply momentum if MIFGSM
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


def load_clean_images(testing_dataset_dir, transform):
    """Load all clean images from testing_dataset"""
    image_files = sorted([f for f in os.listdir(testing_dataset_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('.')[0]))
    
    tensor_list = []
    for img_file in image_files:
        img_path = os.path.join(testing_dataset_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        tensor_list.append(img_tensor)
    
    return tensor_list, image_files


def generate_single_frame_adversarial(temporal_context, frame_idx, model, attack, target):
    """
    Generate adversarial version of a single frame WITH TEMPORAL CONTEXT.
    
    CRITICAL: Must perturb frame within its temporal sequence [f0...fi], not in isolation!
    The SDNN model processes temporal sequences, so gradients computed on a single frame
    in isolation are meaningless and produce zero perturbations.
    
    Args:
        temporal_context: List of clean frames [f0, f1, ..., fi] up to current frame
        frame_idx: Index of the frame to perturb (i)
        model: SDNN model
        attack: torchattacks attack object
        target: Target steering angle value from clean window
    
    Returns:
        Adversarial frame tensor [C, H, W] - only the perturbed fi
    """
    device = next(model.parameters()).device
    
    # Create sequence WITH temporal context [1, C, H, W, T]
    # This gives SDNN proper temporal information for gradient computation
    sequence = torch.stack(temporal_context, dim=3).unsqueeze(0).to(device)
    
    # Generate adversarial version (entire sequence gets perturbed)
    wrapped_attack = RegressionAttackWrapper(attack, model, target)
    adv_seq = wrapped_attack(sequence)
    
    # Extract ONLY the perturbed version of the current frame (fi)
    adv_frame = adv_seq[0, :, :, :, frame_idx]  # [C, H, W]
    
    return adv_frame


def run_inference_on_sequence(sequence_tensor, model):
    """
    Run inference on a sequence tensor.
    
    Args:
        sequence_tensor: [1, C, H, W, T]
        model: SDNN model
    
    Returns:
        Steering angle prediction (float)
    """
    model.eval()
    with torch.no_grad():
        output, _, _ = model(sequence_tensor)
        # SDNN Output layer produces [batch, features, time]
        # Use the LAST timestep (after processing full sequence)
        # This matches Lava inference behavior (reads output after pipeline delay)
        prediction = output[0, 0, -1]
    model.train()
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Single Frame Perturbation Analysis')
    parser.add_argument('--attack', type=str, default='PGD', choices=['FGSM', 'PGD', 'MIFGSM'],
                        help='Attack method (default: PGD)')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Maximum perturbation (default: 0.03)')
    parser.add_argument('--alpha', type=float, default=0.007,
                        help='Step size for iterative attacks (default: 0.007)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for iterative attacks (default: 10)')
    parser.add_argument('--decay', type=float, default=1.0,
                        help='Decay factor for momentum (MIFGSM only, default: 1.0)')
    parser.add_argument('--random_start', action='store_true',
                        help='Use random start for PGD')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Single Frame Perturbation Analysis - {args.attack}")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.attack)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create folder name
    eps_str = f"{args.eps}".rstrip('0').rstrip('.').replace('.', '')
    alpha_str = f"{args.alpha}".rstrip('0').rstrip('.').replace('.', '')
    decay_str = f"{args.decay}".rstrip('0').rstrip('.').replace('.', '')
    
    if args.attack == 'FGSM':
        folder_name = f"single_frame_eps{eps_str}"
    elif args.attack == 'PGD':
        folder_name = f"single_frame_eps{eps_str}_alpha{alpha_str}_steps{args.steps}"
        if args.random_start:
            folder_name += "_randstart"
    elif args.attack == 'MIFGSM':
        folder_name = f"single_frame_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_decay{decay_str}"
    
    results_path = os.path.join(output_dir, f'results_{folder_name}.txt')
    adv_images_dir = os.path.join(output_dir, f'adv_images_{folder_name}')
    checking_dir = os.path.join(output_dir, f'checking_{folder_name}')
    os.makedirs(adv_images_dir, exist_ok=True)
    os.makedirs(checking_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Attack: {args.attack}")
    print(f"  Epsilon: {args.eps}")
    if args.attack in ['PGD', 'MIFGSM']:
        print(f"  Alpha: {args.alpha}")
        print(f"  Steps: {args.steps}")
    if args.attack == 'MIFGSM':
        print(f"  Decay: {args.decay}")
    if args.attack == 'PGD' and args.random_start:
        print(f"  Random Start: True")
    print()
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = Network().to(device)
    loaded_state_dict = torch.load(model_path, map_location=device)
    
    # Filter out mismatched keys (e.g., BatchNorm shape differences)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in loaded_state_dict.items() 
                     if k in model_state and v.size() == model_state[k].size()}
    
    model.load_state_dict(filtered_dict, strict=False)
    model.train()
    print("  Model loaded\n")
    
    # Initialize attack
    print(f"Initializing {args.attack} attack...")
    if args.attack == 'FGSM':
        attack = torchattacks.FGSM(model, eps=args.eps)
    elif args.attack == 'PGD':
        attack = torchattacks.PGD(model, eps=args.eps, alpha=args.alpha, 
                                  steps=args.steps, random_start=args.random_start)
    elif args.attack == 'MIFGSM':
        attack = torchattacks.MIFGSM(model, eps=args.eps, alpha=args.alpha, 
                                     steps=args.steps, decay=args.decay)
    print("  Attack initialized\n")
    
    # Load clean images
    print("Loading clean images from testing_dataset...")
    transform = transforms.Compose([
        transforms.Resize((33, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    clean_frames, image_files = load_clean_images(testing_dataset_dir, transform)
    num_frames = len(clean_frames)
    print(f"  Loaded {num_frames} clean frames\n")
    
    # Main analysis loop - Growing Window Approach
    print(f"Starting single frame perturbation analysis (Growing Window)...")
    print(f"  Iteration 1: [f0 clean] vs [f0 perturbed]")
    print(f"  Iteration 2: [f0, f1 clean] vs [f0 clean, f1 perturbed]")
    print(f"  Iteration 3: [f0, f1, f2 clean] vs [f0, f1 clean, f2 perturbed]")
    print(f"  ... and so on up to {num_frames} frames\n")
    print(f"  This will run {num_frames * 2} inferences total\n")
    
    results = []
    
    # Generate adversarial versions of all frames first (for speed)
    print("Generating adversarial versions of all frames...")
    adv_frames = []
    for frame_idx in tqdm(range(num_frames), desc="Generating adversarial frames"):
        # For the first frame, use a single-frame target
        if frame_idx == 0:
            single_frame_seq = clean_frames[0].unsqueeze(0).unsqueeze(-1).to(device)
            model.eval()
            with torch.no_grad():
                output, _, _ = model(single_frame_seq)
                # SDNN Output: [batch, features, time] - use last timestep
                target = output[0, 0, -1]
            model.train()
        else:
            # For subsequent frames, use target from clean window [0:frame_idx+1]
            clean_window = torch.stack(clean_frames[:frame_idx+1], dim=3).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output, _, _ = model(clean_window)
                # SDNN Output: [batch, features, time] - use last timestep
                target = output[0, 0, -1]
            model.train()
        
        # Generate adversarial version of this frame WITH temporal context
        adv_frame = generate_single_frame_adversarial(
            clean_frames[:frame_idx+1],  # Pass full temporal context [f0...fi]
            frame_idx,                    # Index of frame to extract
            model, 
            attack, 
            target
        )
        adv_frames.append(adv_frame)
        
        # Save adversarial frame
        adv_img = adv_frame * 0.5 + 0.5  # Denormalize
        adv_img = torch.clamp(adv_img, 0, 1)
        adv_img_array = (adv_img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        adv_img_pil = Image.fromarray(adv_img_array)
        adv_img_pil.save(os.path.join(adv_images_dir, image_files[frame_idx]))
    
    print("All adversarial frames generated\n")
    
    # Now run growing window analysis
    print("Running growing window analysis...")
    for i in tqdm(range(num_frames), desc="Analyzing frames"):
        # Iteration i: use frames [0 to i] (i+1 frames total)
        window_size = i + 1
        
        # Create clean window: [f0, f1, ..., fi] all clean
        clean_window = torch.stack(clean_frames[:window_size], dim=3).unsqueeze(0).to(device)
        clean_sa = run_inference_on_sequence(clean_window, model)
        
        # Create perturbed window: [f0, f1, ..., f(i-1) clean, fi perturbed]
        perturbed_window_frames = clean_frames[:window_size].copy()
        perturbed_window_frames[i] = adv_frames[i]
        perturbed_window = torch.stack(perturbed_window_frames, dim=3).unsqueeze(0).to(device)
        perturbed_sa = run_inference_on_sequence(perturbed_window, model)
        
        # Calculate impact
        absolute_diff = perturbed_sa - clean_sa
        percent_diff = (absolute_diff / clean_sa * 100) if clean_sa != 0 else 0
        
        # Store results
        results.append({
            'iteration': i + 1,
            'frame': i,
            'image': image_files[i],
            'window_size': window_size,
            'clean_sa': clean_sa,
            'perturbed_sa': perturbed_sa,
            'absolute_diff': absolute_diff,
            'percent_diff': percent_diff
        })
        
        # Create verification image showing clean vs perturbed windows + perturbation magnitude
        # Save first 20, last 10, and every 10th in between
        save_image = (i < 20) or (i >= num_frames - 10) or (i % 10 == 0)
        
        if save_image:
            # 3 rows: Clean window, Perturbed window, Perturbation magnitude
            fig, axes = plt.subplots(3, window_size, figsize=(3*window_size, 9))
            
            # Handle single frame case (iteration 1)
            if window_size == 1:
                axes = axes.reshape(3, 1)
            
            # Top row: Clean window
            for j in range(window_size):
                clean_img = clean_frames[j] * 0.5 + 0.5  # Denormalize
                clean_img = torch.clamp(clean_img, 0, 1)
                axes[0, j].imshow(clean_img.permute(1, 2, 0).cpu().numpy())
                axes[0, j].set_title(f'Clean f{j}', fontsize=8)
                axes[0, j].axis('off')
            
            # Middle row: Perturbed window (all clean except last frame)
            for j in range(window_size):
                if j == i:
                    # Show perturbed frame
                    pert_img = adv_frames[j] * 0.5 + 0.5  # Denormalize
                    pert_img = torch.clamp(pert_img, 0, 1)
                    axes[1, j].imshow(pert_img.permute(1, 2, 0).cpu().numpy())
                    axes[1, j].set_title(f'PERT f{j}', fontsize=8, color='red', weight='bold')
                    axes[1, j].axis('off')
                else:
                    # Show clean frame
                    clean_img = clean_frames[j] * 0.5 + 0.5
                    clean_img = torch.clamp(clean_img, 0, 1)
                    axes[1, j].imshow(clean_img.permute(1, 2, 0).cpu().numpy())
                    axes[1, j].set_title(f'Clean f{j}', fontsize=8)
                    axes[1, j].axis('off')
            
            # Bottom row: Perturbation magnitude (show diff for perturbed frame, blank for others)
            for j in range(window_size):
                if j == i:
                    # Calculate and show perturbation
                    pert = torch.abs(adv_frames[j] - clean_frames[j])
                    p_min, p_max = pert.min(), pert.max()
                    if p_max > p_min:
                        pert_vis = (pert - p_min) / (p_max - p_min)
                    else:
                        pert_vis = pert
                    axes[2, j].imshow(pert_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
                    axes[2, j].set_title(f'Pert (max={p_max:.4f})', fontsize=8, color='red', weight='bold')
                    axes[2, j].axis('off')
                else:
                    # Blank for clean frames
                    axes[2, j].axis('off')
            
            # Add overall title with steering angles
            fig.suptitle(f'Iteration {i+1} (Window Size {window_size})\n'
                        f'Clean SA: {clean_sa:.4f} | Perturbed SA: {perturbed_sa:.4f} | Diff: {absolute_diff:.4f}',
                        fontsize=10, weight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(checking_dir, f'w{i+1}.jpg'), dpi=100, bbox_inches='tight')
            plt.close()
    
    print(f"\nAnalysis complete!\n")
    
    # Save detailed results (Tab-separated for easy Excel import)
    print(f"Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Single Frame Perturbation Analysis - Growing Window - {args.attack}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Attack: {args.attack}\n")
        f.write(f"  Epsilon: {args.eps}\n")
        if args.attack in ['PGD', 'MIFGSM']:
            f.write(f"  Alpha: {args.alpha}\n")
            f.write(f"  Steps: {args.steps}\n")
        if args.attack == 'MIFGSM':
            f.write(f"  Decay: {args.decay}\n")
        f.write(f"\n")
        
        f.write(f"Approach: Growing Window\n")
        f.write(f"  Iteration i: Compare [f0...fi clean] vs [f0...f(i-1) clean, fi perturbed]\n\n")
        
        f.write("-"*100 + "\n")
        f.write(f"{'Iter':<6}{'Frame':<8}{'Window':<10}{'Image':<15}{'Clean SA':<15}{'Perturbed SA':<15}{'Abs Diff':<15}{'% Diff':<15}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            f.write(f"{r['iteration']:<6}{r['frame']:<8}{r['window_size']:<10}{r['image']:<15}"
                   f"{r['clean_sa']:<15.6f}{r['perturbed_sa']:<15.6f}"
                   f"{r['absolute_diff']:<15.6f}{r['percent_diff']:<15.2f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # Summary statistics
        abs_diffs = [abs(r['absolute_diff']) for r in results]
        f.write("Summary Statistics:\n")
        f.write(f"  Mean absolute difference:    {np.mean(abs_diffs):.6f}\n")
        f.write(f"  Max absolute difference:     {np.max(abs_diffs):.6f}\n")
        f.write(f"  Min absolute difference:     {np.min(abs_diffs):.6f}\n")
        f.write(f"  Std dev absolute difference: {np.std(abs_diffs):.6f}\n\n")
        
        # Most impactful frames
        sorted_by_impact = sorted(results, key=lambda x: abs(x['absolute_diff']), reverse=True)
        f.write("Top 10 Most Impactful Frames (by absolute difference):\n")
        for i, r in enumerate(sorted_by_impact[:10], 1):
            f.write(f"  {i}. Iteration {r['iteration']} (Frame {r['frame']}, {r['image']}, Window={r['window_size']}): "
                   f"{abs(r['absolute_diff']):.6f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    # Also save CSV for easy Excel import
    csv_path = results_path.replace('.txt', '.csv')
    with open(csv_path, 'w') as f:
        f.write("Iteration,Frame,WindowSize,Image,CleanSA,PerturbedSA,AbsDiff,PercentDiff\n")
        for r in results:
            f.write(f"{r['iteration']},{r['frame']},{r['window_size']},{r['image']},"
                   f"{r['clean_sa']:.6f},{r['perturbed_sa']:.6f},"
                   f"{r['absolute_diff']:.6f},{r['percent_diff']:.2f}\n")
    
    print(f"  Results saved!\n")
    
    # Print summary to console
    abs_diffs = [abs(r['absolute_diff']) for r in results]
    sorted_by_impact = sorted(results, key=lambda x: abs(x['absolute_diff']), reverse=True)
    print(f"{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(f"Total frames analyzed:       {num_frames}")
    print(f"Total inferences run:        {num_frames * 2}")
    print(f"Mean absolute impact:        {np.mean(abs_diffs):.6f}")
    print(f"Max absolute impact:         {np.max(abs_diffs):.6f} (Iteration {sorted_by_impact[0]['iteration']}, Frame {sorted_by_impact[0]['frame']})")
    print(f"Min absolute impact:         {np.min(abs_diffs):.6f} (Iteration {sorted_by_impact[-1]['iteration']}, Frame {sorted_by_impact[-1]['frame']})")
    print(f"\nAdversarial images saved to: {adv_images_dir}/")
    print(f"Verification images:         {checking_dir}/")
    print(f"Detailed results saved to:   {results_path}")
    print(f"CSV file (for Excel):        {csv_path}")
    print(f"\nCheck {checking_dir}/ to verify correct windows are being used!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
