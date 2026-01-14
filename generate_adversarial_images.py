"""
Generate Adversarial Images for PilotNet from testing_dataset

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import lava.lib.dl.slayer as slayer
from Attacks.attacks import FGSM, PGD, MIFGSM


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
        for block in self.blocks:
            x = block(x)
        return x, None, None


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial images for PilotNet')
    parser.add_argument('--attack', type=str, required=True, choices=['FGSM', 'PGD', 'MIFGSM'],
                        help='Type of attack to perform')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Epsilon (perturbation bound)')
    parser.add_argument('--alpha', type=float, default=0.007,
                        help='Step size for iterative attacks')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for iterative attacks')
    parser.add_argument('--decay', type=float, default=1.0,
                        help='Decay factor for MIFGSM')
    
    args = parser.parse_args()
    
    # Create folder name based on attack parameters
    if args.attack == 'FGSM':
        folder_name = f"adv_img_eps{args.eps}"
    elif args.attack == 'PGD':
        folder_name = f"adv_img_eps{args.eps}_alpha{args.alpha}_steps{args.steps}"
    elif args.attack == 'MIFGSM':
        folder_name = f"adv_img_eps{args.eps}_alpha{args.alpha}_steps{args.steps}_decay{args.decay}"
    
    print(f"\n{'='*60}")
    print(f"Generate Adversarial Images - {args.attack}")
    print(f"{'='*60}\n")
    
    # Paths
    base_dir = os.path.dirname(__file__)
    testing_dir = os.path.join(base_dir, 'testing_dataset')
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    attack_dir = os.path.join(base_dir, args.attack)
    adv_images_dir = os.path.join(attack_dir, folder_name)
    
    # Create directories (attack directory and adversarial images subdirectory)
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(adv_images_dir, exist_ok=True)
    
    # Load model
    print("Loading model")
    model = Network()
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.train()  # IMPORTANT: Use train mode for gradient computation in attacks
    print("Model loaded\n")
    
    # Load images
    print("Loading images from testing_dataset")
    all_files = os.listdir(testing_dir)
    frame_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    frame_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    transform = transforms.Compose([
        transforms.Resize((33, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    tensor_list = [transform(Image.open(os.path.join(testing_dir, f)).convert('RGB')) 
                   for f in frame_files]
    sequence_tensor = torch.stack(tensor_list, dim=3).unsqueeze(0)  # Add batch dim
    print(f"Loaded {len(frame_files)} images\n")
    
    # Initialize attack
    print(f"Initializing {args.attack} attack")
    if args.attack == 'FGSM':
        attack = FGSM(model, eps=args.eps)
    elif args.attack == 'PGD':
        attack = PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif args.attack == 'MIFGSM':
        attack = MIFGSM(model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay)
    
    # Generate adversarial images
    print(f"Generating adversarial images")
    # Get target output (no grad needed here)
    with torch.no_grad():
        output, _, _ = model(sequence_tensor)
        target = output.mean()
    
    print(f"  Target value: {target.item():.6f}")
    print(f"  Sequence tensor shape: {sequence_tensor.shape}")
    print(f"  Sequence tensor requires_grad: {sequence_tensor.requires_grad}")
    
    # Generate adversarial images (gradients ARE needed here)
    adv_sequence = attack(sequence_tensor, target)
    
    # Check if perturbation was actually applied
    diff = torch.abs(adv_sequence - sequence_tensor)
    print(f"  Perturbation applied - min: {diff.min().item():.6f}, max: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    
    print(f"Generated adversarial sequence\n")
    
    # Save adversarial images
    print(f"Saving adversarial images to {args.attack}/{folder_name}/")
    adv_imgs = adv_sequence.squeeze(0)  # Remove batch dim
    orig_imgs = sequence_tensor.squeeze(0)  # Original images
    
    for i, filename in enumerate(frame_files):
        adv_frame = adv_imgs[:, :, :, i]
        img_tensor = adv_frame * 0.5 + 0.5  # Denormalize
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img.save(os.path.join(adv_images_dir, filename))
    
    # Calculate perturbation statistics
    perturbation_mean = torch.mean(torch.abs(adv_sequence - sequence_tensor)).item()
    perturbation_max = torch.max(torch.abs(adv_sequence - sequence_tensor)).item()
    
    # Create verification images (first 5 + 5 random samples)
    print(f"Creating verification images")
    verification_dir = os.path.join(attack_dir, f'verification_{folder_name}')
    os.makedirs(verification_dir, exist_ok=True)
    
    # Select indices: first 5 + 5 random
    import random
    indices = list(range(min(5, len(frame_files))))  # First 5
    if len(frame_files) > 5:
        remaining = list(range(5, len(frame_files)))
        random_indices = random.sample(remaining, min(5, len(remaining)))
        indices.extend(random_indices)
    
    for idx, i in enumerate(indices):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        orig_img = orig_imgs[:, :, :, i] * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0].imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Adversarial image
        adv_img = adv_imgs[:, :, :, i] * 0.5 + 0.5
        adv_img = torch.clamp(adv_img, 0, 1)
        axes[1].imshow(adv_img.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title('Adversarial')
        axes[1].axis('off')
        
        # Perturbation (normalized to full range for visibility)
        perturbation = torch.abs(adv_imgs[:, :, :, i] - orig_imgs[:, :, :, i])
        # Normalize to [0, 1] range for better visibility
        p_min = perturbation.min()
        p_max = perturbation.max()
        if p_max > p_min:
            perturbation_vis = (perturbation - p_min) / (p_max - p_min)
        else:
            perturbation_vis = perturbation
        axes[2].imshow(perturbation_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[2].set_title(f'Perturbation (max={p_max.item():.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(verification_dir, f'verify_{idx+1:02d}_{frame_files[i]}'), dpi=150)
        plt.close()
    
    print(f"Saved {len(frame_files)} adversarial images")
    print(f"Created {len(indices)} verification images in {args.attack}/verification_{folder_name}/\n")
    # print(f"{'='*60}")
    # print(f"Perturbation Statistics:")
    # print(f"  Mean absolute perturbation: {perturbation_mean:.6f}")
    # print(f"  Max absolute perturbation:  {perturbation_max:.6f}")
    # print(f"  Epsilon (bound):            {args.eps}")
    # print(f"\nAdversarial images saved to: {args.attack}/{folder_name}/")
    # print(f"\nNext step: Run inference on these images using:")
    # print(f"  python run_inference_on_adversarial.py --attack {args.attack} --folder {folder_name}")
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
