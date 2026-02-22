"""
Generate Adversarial Frames with Temporal Context for Single Frame Analysis

This script generates adversarial versions of each frame where each frame is perturbed
within its growing temporal context. Frame i is perturbed within context [f0...fi].

After generation, use analyze_single_frame_impact.py to run the windowed analysis.
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
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


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
    sequence = torch.stack(temporal_context, dim=3).unsqueeze(0).to(device)
    
    # Generate adversarial version (entire sequence gets perturbed)
    wrapped_attack = RegressionAttackWrapper(attack, model, target)
    adv_seq = wrapped_attack(sequence)
    
    # Extract ONLY the perturbed version of the current frame (fi)
    adv_frame = adv_seq[0, :, :, :, frame_idx]  # [C, H, W]
    
    return adv_frame


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial frames with temporal context')
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
    print(f"Generate Single Frame Adversarial Images - {args.attack}")
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
    
    adv_images_dir = os.path.join(output_dir, f'adv_images_{folder_name}')
    verification_dir = os.path.join(output_dir, f'verification_{folder_name}')
    os.makedirs(adv_images_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
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
    
    # Filter out mismatched keys
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
    
    print(f"Generating adversarial frames with temporal context...")
    print(f"  Frame i will be perturbed within context [f0...fi]")
    print(f"  This takes time (~{num_frames * 10} forward/backward passes)\n")
    
    adv_frames = []
    for frame_idx in tqdm(range(num_frames), desc="Generating adversarial frames"):
        # For the first frame, use a single-frame target
        if frame_idx == 0:
            single_frame_seq = clean_frames[0].unsqueeze(0).unsqueeze(-1).to(device)
            model.eval()
            with torch.no_grad():
                output, _, _ = model(single_frame_seq)
                target = output[0, 0, -1]  # Last timestep
            model.train()
        else:
            # For subsequent frames, use target from clean window [0:frame_idx+1]
            clean_window = torch.stack(clean_frames[:frame_idx+1], dim=3).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output, _, _ = model(clean_window)
                target = output[0, 0, -1]  # Last timestep
            model.train()
        
        # Generate adversarial version of this frame WITH temporal context
        adv_frame = generate_single_frame_adversarial(
            clean_frames[:frame_idx+1],  # Full temporal context [f0...fi]
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
    
    print("\nAll adversarial frames generated!\n")
    
    # Create verification images (first 10, last 10, every 10th)
    print("Creating verification images...")
    verification_indices = list(range(10)) + list(range(num_frames-10, num_frames)) + list(range(10, num_frames-10, 10))
    verification_indices = sorted(set(verification_indices))
    
    for idx in tqdm(verification_indices, desc="Creating verification images"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        clean_img = clean_frames[idx] * 0.5 + 0.5
        clean_img = torch.clamp(clean_img, 0, 1)
        axes[0].imshow(clean_img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Adversarial
        adv_img = adv_frames[idx] * 0.5 + 0.5
        adv_img = torch.clamp(adv_img, 0, 1)
        axes[1].imshow(adv_img.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title('Adversarial')
        axes[1].axis('off')
        
        # Perturbation
        pert = torch.abs(adv_frames[idx] - clean_frames[idx])
        p_min, p_max = pert.min(), pert.max()
        if p_max > p_min:
            pert_vis = (pert - p_min) / (p_max - p_min)
        else:
            pert_vis = pert
        axes[2].imshow(pert_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[2].set_title(f'Perturbation (max={p_max:.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(verification_dir, f'verification_{image_files[idx]}'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n{'='*70}")
    print("Generation Complete!")
    print(f"{'='*70}")
    print(f"Adversarial images saved to: {adv_images_dir}/")
    print(f"Verification images:         {verification_dir}/")
    print(f"\nNext step: Run windowed analysis using:")
    print(f"  py -3.9 Using_torchattacks/sa_compare/analyze_single_frame_impact.py \\")
    print(f"    --attack {args.attack} --folder {folder_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
