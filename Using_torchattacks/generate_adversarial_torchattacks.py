"""
Generate Adversarial Images for PilotNet from testing_dataset using torchattacks library

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
        """
        Simplified forward for attack generation.
        We don't need event_cost and count during attacks.
        """
        for block in self.blocks:
            x = block(x)
        return x, None, None


class RegressionAttackWrapper:
    """
    Wrapper to adapt torchattacks for regression tasks.
    
    Since torchattacks hardcodes CrossEntropyLoss inside forward() methods,
    we must reimplement the attack logic using MSELoss for regression.
    This wrapper reads attack parameters from torchattacks objects but
    implements the actual attack algorithm for regression models.
    
    Supports: FGSM, PGD, MIFGSM, and other gradient-based attacks
    """
    def __init__(self, attack, model, target_value):
        self.attack = attack
        self.model = model
        self.target_value = target_value
        self.device = next(model.parameters()).device
        
        # Extract attack parameters from torchattacks object
        self.attack_name = self.attack.attack
        self.eps = self.attack.eps
        
        # Iterative attack parameters
        self.steps = getattr(self.attack, 'steps', 1)
        self.alpha = getattr(self.attack, 'alpha', self.eps)
        self.random_start = getattr(self.attack, 'random_start', False)
        
        # Momentum-based attack parameters (MIFGSM)
        self.decay = getattr(self.attack, 'decay', None)
        
    def __call__(self, images):
        """
        Generate adversarial images for regression task.
        
        Args:
            images: Input images tensor [N, C, H, W, T]
        
        Returns:
            Adversarial images
        """
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
            
            # MSE loss for regression - NEGATE to maximize distance (untargeted attack)
            # This makes the adversarial output DIFFERENT from the original prediction
            cost = -nn.MSELoss()(final_prediction, target.squeeze())
            
            # Backward pass to get gradients w.r.t. input
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
            
            # Project perturbation to epsilon ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
        
        return adv_images


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial images using torchattacks')
    parser.add_argument('--attack', type=str, required=True, choices=['FGSM', 'PGD', 'MIFGSM'],
                        help='Attack method')
    parser.add_argument('--eps', type=float, default=8/255,
                        help='Maximum perturbation (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='Step size for iterative attacks (default: 2/255)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for iterative attacks (default: 10)')
    parser.add_argument('--decay', type=float, default=1.0,
                        help='Decay factor for momentum (MIFGSM only, default: 1.0)')
    parser.add_argument('--random_start', action='store_true',
                        help='Use random start for PGD')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Generate Adversarial Images - {args.attack} (torchattacks)")
    print(f"{'='*60}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    # Create output directory structure
    output_base = os.path.join(os.path.dirname(__file__), args.attack)
    os.makedirs(output_base, exist_ok=True)
    
    # Create folder name based on parameters
    # Remove trailing zeros
    eps_str = f"{args.eps}".rstrip('0').rstrip('.').replace('.', '')
    alpha_str = f"{args.alpha}".rstrip('0').rstrip('.').replace('.', '')
    decay_str = f"{args.decay}".rstrip('0').rstrip('.').replace('.', '')
    
    if args.attack == 'FGSM':
        folder_name = f"adv_img_eps{eps_str}"
    elif args.attack == 'PGD':
        folder_name = f"adv_img_eps{eps_str}_alpha{alpha_str}_steps{args.steps}"
        if args.random_start:
            folder_name += "_randstart"
    elif args.attack == 'MIFGSM':
        folder_name = f"adv_img_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_decay{decay_str}"
    
    output_dir = os.path.join(output_base, folder_name)
    verification_dir = os.path.join(output_base, f'verification_{folder_name}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
    # Load model
    print("Loading model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)
    
    # Load checkpoint - filter out mismatched keys (e.g., BatchNorm shape differences)
    loaded_state_dict = torch.load(model_path, map_location=device)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in loaded_state_dict.items() 
                     if k in model_state and v.size() == model_state[k].size()}
    
    model.load_state_dict(filtered_dict, strict=False)
    model.train()  # Important for gradient computation
    print("Model loaded\n")
    
    # Initialize torchattacks attack
    print(f"Initializing {args.attack} attack from torchattacks")
    if args.attack == 'FGSM':
        attack = torchattacks.FGSM(model, eps=args.eps)
    elif args.attack == 'PGD':
        attack = torchattacks.PGD(model, eps=args.eps, alpha=args.alpha, 
                                  steps=args.steps, random_start=args.random_start)
    elif args.attack == 'MIFGSM':
        attack = torchattacks.MIFGSM(model, eps=args.eps, alpha=args.alpha, 
                                     steps=args.steps, decay=args.decay)
    
    print("Attack initialized\n")
    
    # Load and preprocess images (SAME AS CUSTOM IMPLEMENTATION)
    print("Loading images from testing_dataset")
    image_files = sorted([f for f in os.listdir(testing_dataset_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('.')[0]))
    
    # Image transformations (normalized to [-1, 1] like training)
    transform_to_tensor = transforms.Compose([
        transforms.Resize((33, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    tensor_list = []
    original_images = []
    
    for img_file in image_files:
        img_path = os.path.join(testing_dataset_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform_to_tensor(img)
        tensor_list.append(img_tensor)
        original_images.append(img_tensor.clone())
    
    if len(tensor_list) == 0:
        print("No images found in testing_dataset")
        return
    
    print(f"Loaded {len(tensor_list)} images\n")
    
    # Stack into sequence tensor [N, C, H, W, T]
    sequence_tensor = torch.stack(tensor_list, dim=3).unsqueeze(0).to(device)
    print(f"  Sequence tensor shape: {sequence_tensor.shape}")
    
    # Get target value (use eval mode for deterministic prediction)
    model.eval()  # Disable dropout for consistent target
    with torch.no_grad():
        output, _, _ = model(sequence_tensor)
        target = output.mean().item()
    model.train()  # Re-enable dropout for attack gradient computation
    print(f"  Target value: {target}\n")
    
    # Generate adversarial images using wrapped attack
    print("Generating adversarial images")
    wrapped_attack = RegressionAttackWrapper(attack, model, target)
    adv_sequence = wrapped_attack(sequence_tensor)
    print("Generated adversarial sequence\n")
    
    # Calculate perturbation statistics
    perturbation = torch.abs(adv_sequence - sequence_tensor)
    print(f"Perturbation Statistics:")
    print(f"  Mean absolute perturbation: {perturbation.mean():.6f}")
    print(f"  Max absolute perturbation:  {perturbation.max():.6f}")
    print(f"  Epsilon (bound):            {args.eps}\n")
    
    # Save adversarial images
    print(f"Saving adversarial images to {output_dir}/")
    
    for i, img_file in enumerate(image_files):
        adv_img = adv_sequence[0, :, :, :, i]
        # Denormalize from [-1, 1] to [0, 1]
        img_tensor = adv_img * 0.5 + 0.5
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array)
        output_path = os.path.join(output_dir, img_file)
        img.save(output_path)
    
    print(f"Saved {len(image_files)} adversarial images\n")
    
    # Create verification images for all frames in ascending order
    print("Creating verification images for all frames")
    
    for idx, i in enumerate(range(len(image_files))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image (denormalized)
        orig_img = original_images[i] * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0].imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Adversarial image (denormalized)
        adv_img = adv_sequence[0, :, :, :, i] * 0.5 + 0.5
        adv_img = torch.clamp(adv_img, 0, 1)
        axes[1].imshow(adv_img.permute(1, 2, 0).cpu().detach().numpy())
        axes[1].set_title('Adversarial')
        axes[1].axis('off')
        
        # Perturbation
        pert = torch.abs(adv_sequence[0, :, :, :, i] - original_images[i].to(device))
        p_min, p_max = pert.min(), pert.max()
        if p_max > p_min:
            pert_vis = (pert - p_min) / (p_max - p_min)
        else:
            pert_vis = pert
        
        axes[2].imshow(pert_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[2].set_title(f'Perturbation (max={p_max:.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(verification_dir, f'verification_{image_files[i]}'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Created {len(image_files)} verification images in {verification_dir}/\n")
    
    # print(f"{'='*60}")
    # print(f"Adversarial images saved to: {output_dir}/")
    # print(f"\nNext step: Run inference on these images using:")
    # print(f"  python Using_torchattacks/run_inference_torchattacks.py --attack {args.attack} --folder {folder_name}")
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    main()