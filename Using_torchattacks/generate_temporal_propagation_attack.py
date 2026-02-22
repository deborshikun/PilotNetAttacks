"""
Generate Temporal Propagation Adversarial Attack (NON-OVERLAPPING)

This script implements a novel temporal attack approach:
1. Divide sequence into non-overlapping windows of (window_size + 1) frames
2. For each window: perturb first 'window_size' frames, keep last frame clean
3. Pattern: [adv, adv, adv, adv, adv, CLEAN] repeated
4. Measure how adversarial history affects the prediction on the clean future frame
5. Save the complete mixed sequence for Lava SDNN validation

Key Difference:
- Sliding window: Attack all frames, measure MSE on perturbed frames
- Overlapping temporal: Attack frames [i, i+1, ..., i+4], test clean frame i+5 (frames reused)
- This approach (non-overlapping): Attack frames [0-4], clean 5, attack [6-10], clean 11, etc.

Example with window_size=5:
- Window 0: Frames [0,1,2,3,4] adversarial, frame 5 clean
- Window 1: Frames [6,7,8,9,10] adversarial, frame 11 clean
- Saved sequence: 0_adv, 1_adv, 2_adv, 3_adv, 4_adv, 5_clean, 6_adv, 7_adv, ...

This tests temporal attack propagation in SDNNs where delta/sigma states carry over.
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
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lava.lib.dl.slayer as slayer
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.lib.dl import netx

# Add pilotnet_sdnn to path for inference utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pilotnet_sdnn'))
from dataset import PilotNetDataset
from utils import PilotNetEncoder, PilotNetDecoder, CustomSimRunConfig, get_input_transform


def load_ground_truth_and_predictions(results_path):
    """Load ground truth labels and original predictions from pilotnet_sdnn/results.txt"""
    ground_truth = {}
    original_predictions = {}
    with open(results_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                img_name = parts[0]
                gt_value = float(parts[1])
                orig_pred = float(parts[2])
                ground_truth[img_name] = gt_value
                original_predictions[img_name] = orig_pred
    return ground_truth, original_predictions


def run_lava_inference_on_sequence(lava_net, sequence_tensor, net_config, temp_dir_base):
    """
    Run Lava SDNN inference on a sequence tensor.
    Uses the same approach as run_inference_torchattacks.py - saves frames to temp directory
    and uses PilotNetDataset for proper format compatibility.
    
    Args:
        lava_net: Lava SDNN network
        sequence_tensor: Tensor of shape [1, C, H, W, T] (normalized -1 to 1)
        net_config: Network configuration
        temp_dir_base: Base directory for temporary files
    
    Returns:
        predictions: Array of predictions for each timestep
    """
    T = sequence_tensor.shape[-1]
    num_steps = T + len(lava_net.layers)
    out_offset = len(lava_net.layers) + 3
    
    # Create temporary directory structure
    temp_data_dir = os.path.join(temp_dir_base, 'temp_inference_data', 'driving_dataset')
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Save frames to temporary directory
    frame_files = []
    for t in range(T):
        frame = sequence_tensor[0, :, :, :, t].cpu()
        # Denormalize from [-1, 1] to [0, 1]
        frame = frame * 0.5 + 0.5
        frame = torch.clamp(frame, 0, 1)
        # Convert to PIL Image and save
        frame_array = (frame.permute(1, 2, 0).numpy() * 255).astype('uint8')
        frame_img = Image.fromarray(frame_array)
        frame_filename = f"frame_{t:04d}.jpg"
        frame_img.save(os.path.join(temp_data_dir, frame_filename))
        frame_files.append(frame_filename)
    
    # Create data.txt for dataset
    data_txt_path = os.path.join(temp_data_dir, 'data.txt')
    with open(data_txt_path, 'w') as f:
        for frame_file in frame_files:
            f.write(f"{frame_file} 0.0\n")  # Dummy angle
    
    # Setup dataset using PilotNetDataset
    transform = get_input_transform(net_config)
    temp_data_parent = os.path.join(temp_dir_base, 'temp_inference_data')
    
    dataset = PilotNetDataset(
        path=temp_data_parent,
        size=lava_net.inp.shape[:2],
        transform=transform,
        visualize=True,
        sample_offset=0
    )
    
    # Setup encoder/decoder
    input_encoder = PilotNetEncoder(
        shape=lava_net.inp.shape,
        net_config=net_config,
        compression=io.encoder.Compression.DENSE
    )
    output_decoder = PilotNetDecoder(shape=lava_net.out.shape)
    output_logger = io.sink.RingBuffer(shape=lava_net.out_layer.shape, buffer=num_steps)
    
    dataloader = io.dataloader.SpikeDataloader(dataset=dataset)
    
    # Connect processes
    dataloader.s_out.connect(input_encoder.inp)
    input_encoder.out.connect(lava_net.inp)
    lava_net.out.connect(output_decoder.inp)
    output_decoder.out.connect(output_logger.a_in)
    
    # Run inference
    run_config = CustomSimRunConfig()
    lava_net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
    
    output = output_logger.data.get().flatten()
    lava_net.stop()
    
    # Extract predictions (skip offset)
    predictions = output[out_offset:out_offset + T]
    
    # Cleanup temporary directory
    shutil.rmtree(os.path.join(temp_dir_base, 'temp_inference_data'))
    
    return predictions


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
            final_prediction = outputs.mean()
            
            # MSE loss - NEGATE to maximize distance (untargeted attack)
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


def temporal_propagation_attack(pytorch_model, lava_net, net_config, original_images, attack_params, ground_truth_labels, original_predictions, output_dir, folder_name, window_size=5):
    """
    Generate adversarial examples using temporal propagation approach (NON-OVERLAPPING).
    
    Uses PyTorch model for adversarial generation (gradient-based attacks)
    Uses Lava SDNN for inference/evaluation (same as deployment)
    
    PROCEDURE:
    ==========
    1. Run Lava SDNN inference on entire original sequence (get baseline predictions)
    2. For each window:
       - Use Lava original prediction as target value
       - Generate adversarial perturbations using PyTorch (requires gradients)
       - Build mixed sequence: window_size adversarial frames + 1 clean frame
    3. Run Lava SDNN inference on complete mixed adversarial/clean sequence
    4. Compare predictions on clean frames: original vs with adversarial history
    
    This ensures all predictions use Lava SDNN (sound procedure) while keeping
    PyTorch for gradient-based attack generation (practical requirement).
    
    EXPLANATION:
    ============
    Instead of measuring MSE on perturbed frames, we:
    1. Divide the sequence into non-overlapping windows of size (window_size + 1)
    2. For each window: perturb first 'window_size' frames, keep last frame clean
    3. Pattern: [adv, adv, adv, adv, adv, CLEAN] repeated throughout sequence
    4. Run Lava SDNN inference on mixed sequence to test how adversarial history affects clean frames
    5. Save the complete mixed sequence for repeated validation
    
    This measures how adversarial temporal history affects predictions on clean future frames.
    
    Example with 200 frames and window_size=5:
    - Window 0: Frames [0,1,2,3,4] adversarial, frame 5 CLEAN
    - Window 1: Frames [6,7,8,9,10] adversarial, frame 11 CLEAN
    - Window 2: Frames [12,13,14,15,16] adversarial, frame 17 CLEAN
    - ...
    - Total windows: 200 // 6 = 33 complete windows
    
    Saved sequence: 0_adv, 1_adv, 2_adv, 3_adv, 4_adv, 5_clean, 6_adv, 7_adv, ...
    
    Args:
        model: SDNN model
        original_images: Original image tensor [1, C, H, W, T]
        attack_params: Dictionary with attack configuration
        window_size: Size of temporal window to perturb (default: 5)
    
    Returns:
        results: List of dictionaries with window results
    """
    device = next(pytorch_model.parameters()).device
    T = original_images.shape[-1]  # Total number of frames
    
    # Non-overlapping windows: each window is [window_size perturbed frames] + [1 clean frame]
    frames_per_window = window_size + 1  # e.g., 5 perturbed + 1 clean = 6 frames per window
    num_windows = T // frames_per_window  # How many complete windows fit in the sequence
    
    # Create directories for saving adversarial images
    adv_images_dir = os.path.join(output_dir, 'adversarial_windows')
    verification_dir = os.path.join(output_dir, 'verification')
    os.makedirs(adv_images_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"TEMPORAL PROPAGATION ADVERSARIAL ATTACK (NON-OVERLAPPING)")
    print(f"{'='*70}")
    print(f"Total frames: {T}")
    print(f"Window size (perturbed): {window_size}")
    print(f"Frames per window (perturbed + clean): {frames_per_window}")
    print(f"Number of test windows: {num_windows}")
    print(f"Pattern: [adv, adv, adv, adv, adv, CLEAN] repeated")
    print(f"{'='*70}\n")
    
    # Step 1: Run Lava SDNN inference on original sequence
    print("Step 1: Running Lava SDNN inference on original sequence")
    print(f"{'='*70}")
    lava_original_predictions = run_lava_inference_on_sequence(lava_net, original_images, net_config, output_dir)
    print(f"Got {len(lava_original_predictions)} predictions from Lava SDNN")
    print(f"Sample predictions: {lava_original_predictions[:5]}")
    print(f"{'='*70}\n")
    
    results = []
    
    # Create mixed sequence: adversarial frames followed by clean frames
    mixed_sequence = original_images.clone()
    
    # Step 2: Generate adversarial windows
    print("Step 2: Generating adversarial windows")
    print(f"{'='*70}")
    
    # Process each non-overlapping window
    for win_idx in range(num_windows):
        # Each window covers: [win_idx * frames_per_window] to [(win_idx + 1) * frames_per_window - 1]
        window_start = win_idx * frames_per_window
        window_end = window_start + window_size  # First 'window_size' frames to perturb
        clean_frame_idx = window_start + window_size  # The clean frame right after
        
        print(f"Window {win_idx + 1}/{num_windows}: Frames [{window_start}-{window_end-1}] adversarial, frame {clean_frame_idx} CLEAN")
        
        # Step 2a: Extract window to perturb (frames to attack)
        window_to_perturb = original_images[:, :, :, :, window_start:window_end].clone().to(device)
        
        # Step 2b: Get target value from Lava original predictions (use mean of window)
        target_value = float(lava_original_predictions[window_start:window_end].mean())
        
        # Step 2c: Create attack for THIS window (using PyTorch for gradient computation)
        if attack_params['attack'] == 'FGSM':
            attack = torchattacks.FGSM(pytorch_model, eps=attack_params['eps'])
        elif attack_params['attack'] == 'PGD':
            attack = torchattacks.PGD(pytorch_model, eps=attack_params['eps'], 
                                     alpha=attack_params['alpha'],
                                     steps=attack_params['steps'],
                                     random_start=attack_params.get('random_start', False))
        elif attack_params['attack'] == 'MIFGSM':
            attack = torchattacks.MIFGSM(pytorch_model, eps=attack_params['eps'],
                                        alpha=attack_params['alpha'],
                                        steps=attack_params['steps'],
                                        decay=attack_params['decay'])
        
        wrapped_attack = RegressionAttackWrapper(attack, pytorch_model, target_value)
        
        # Step 2d: Generate adversarial version of THIS window
        adv_window = wrapped_attack(window_to_perturb)
        
        # Update mixed sequence: replace perturbed frames with adversarial versions
        # KEEP the clean frame at clean_frame_idx unchanged (it stays clean)
        mixed_sequence[:, :, :, :, window_start:window_end] = adv_window
        
        # Save adversarial images from this window (save every 10th window + first/last)
        if win_idx % 10 == 0 or win_idx == 0 or win_idx == num_windows - 1:
            window_dir = os.path.join(adv_images_dir, f'window_{win_idx:04d}')
            os.makedirs(window_dir, exist_ok=True)
            
            for frame_idx in range(window_size):
                # Save adversarial frame
                adv_frame = adv_window[0, :, :, :, frame_idx]
                adv_frame_denorm = adv_frame * 0.5 + 0.5
                adv_frame_denorm = torch.clamp(adv_frame_denorm, 0, 1)
                adv_array = (adv_frame_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                adv_img = Image.fromarray(adv_array)
                adv_img.save(os.path.join(window_dir, f'adv_frame_{window_start + frame_idx}.jpg'))
                
                # Create verification image (original vs adversarial vs perturbation)
                orig_frame = window_to_perturb[0, :, :, :, frame_idx]
                orig_frame_denorm = orig_frame * 0.5 + 0.5
                orig_frame_denorm = torch.clamp(orig_frame_denorm, 0, 1)
                
                pert = torch.abs(adv_frame - orig_frame)
                p_min, p_max = pert.min(), pert.max()
                if p_max > p_min:
                    pert_vis = (pert - p_min) / (p_max - p_min)
                else:
                    pert_vis = pert
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(orig_frame_denorm.permute(1, 2, 0).cpu().numpy())
                axes[0].set_title(f'Original Frame {window_start + frame_idx}')
                axes[0].axis('off')
                
                axes[1].imshow(adv_frame_denorm.permute(1, 2, 0).cpu().numpy())
                axes[1].set_title('Adversarial')
                axes[1].axis('off')
                
                axes[2].imshow(pert_vis.permute(1, 2, 0).cpu().numpy(), cmap='hot')
                axes[2].set_title(f'Perturbation (max={p_max:.4f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(verification_dir, f'window_{win_idx:04d}_frame_{window_start + frame_idx}.jpg'),
                           dpi=100, bbox_inches='tight')
                plt.close()
        
        # Store window information (will get predictions after running Lava inference on full sequence)
        pert_norm = torch.norm(adv_window - window_to_perturb).item()
        
        # Storetemp info - will calculate predictions later
        results.append({
            'window_idx': win_idx,
            'perturbed_frames': (window_start, window_end - 1),
            'clean_frame': clean_frame_idx,
            'perturbation_norm': pert_norm
        })
    
    print(f"Generated {num_windows} adversarial windows")
    print(f"{'='*70}\n")
    
    # Step 3: Run Lava SDNN inference on complete mixed sequence
    print("Step 3: Running Lava SDNN inference on mixed adversarial/clean sequence")
    print(f"{'='*70}")
    
    # Reload Lava network (previous one was stopped)
    import os as os_module
    current_dir = os_module.getcwd()
    pilotnet_dir = os_module.path.dirname(os_module.path.dirname(__file__))
    pilotnet_dir = os_module.path.join(pilotnet_dir, 'pilotnet_sdnn')
    os_module.chdir(pilotnet_dir)
    lava_net_mixed = netx.hdf5.Network(net_config='network.net', skip_layers=1)
    os_module.chdir(current_dir)
    
    lava_mixed_predictions = run_lava_inference_on_sequence(lava_net_mixed, mixed_sequence, net_config, output_dir)
    print(f"Got {len(lava_mixed_predictions)} predictions from Lava SDNN on mixed sequence")
    print(f"{'='*70}\n")
    
    # Step 4: Calculate results for each window
    print("Step 4: Analyzing results")
    print(f"{'='*70}")
    for r in results:
        clean_frame_idx = r['clean_frame']
        clean_frame_name = f"{10550 + clean_frame_idx}.jpg"
        
        # Get values
        gt_value = ground_truth_labels.get(clean_frame_name, 0.0)
        lava_original_pred = float(lava_original_predictions[clean_frame_idx])
        lava_mixed_pred = float(lava_mixed_predictions[clean_frame_idx])
        
        # Calculate differences
        prediction_diff = abs(lava_mixed_pred - lava_original_pred)
        orig_error = abs(lava_original_pred - gt_value)
        adv_error = abs(lava_mixed_pred - gt_value)
        
        # Update results dictionary
        r['ground_truth'] = gt_value
        r['lava_original_prediction'] = lava_original_pred
        r['lava_mixed_prediction'] = lava_mixed_pred
        r['prediction_difference'] = prediction_diff
        r['original_error'] = orig_error
        r['adversarial_error'] = adv_error
        
        print(f"Window {r['window_idx'] + 1}: Clean frame {clean_frame_idx}")
        print(f"  GT: {gt_value:.6f} | Lava Original: {lava_original_pred:.6f} | "
              f"Lava Mixed: {lava_mixed_pred:.6f} | Diff: {prediction_diff:.6f}")
    
    print(f"{'='*70}\n")
    
    # Save mixed sequence as flat folder of images for run_inference
    print(f"\n{'='*70}")
    print("Saving mixed adversarial/clean sequence for Lava inference validation")
    print(f"{'='*70}")
    print(f"Pattern: {window_size} adversarial frames followed by 1 clean frame, repeated")
    print(f"This tests how adversarial history affects predictions on clean frames")
    print(f"{'='*70}")
    
    final_adv_images_dir = os.path.join(output_dir, 'adversarial_sequence')
    os.makedirs(final_adv_images_dir, exist_ok=True)
    
    # Denormalize and save each frame from the mixed sequence
    for frame_idx in range(T):
        frame = mixed_sequence[0, :, :, :, frame_idx]
        frame_denorm = frame * 0.5 + 0.5
        frame_denorm = torch.clamp(frame_denorm, 0, 1)
        frame_array = (frame_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        frame_img = Image.fromarray(frame_array)
        
        # Save with same naming as original testing_dataset (10550-10746)
        img_name = f"{10550 + frame_idx}.jpg"
        frame_img.save(os.path.join(final_adv_images_dir, img_name))
        
        # Mark which frames are clean vs adversarial
        window_idx = frame_idx // frames_per_window
        position_in_window = frame_idx % frames_per_window
        frame_type = "CLEAN" if position_in_window == window_size else "ADV"
        if frame_idx < 10 or frame_idx % 20 == 0:
            print(f"  Frame {frame_idx} (img {10550 + frame_idx}.jpg): {frame_type}")
    
    print(f"\nSaved {T} frames to: {final_adv_images_dir}")
    print(f"\nTo run Lava SDNN inference validation on this mixed sequence, use:")
    attack_name = attack_params['attack']
    print(f"  py -3.9 Using_torchattacks/run_inference_torchattacks.py \\")
    print(f"    --attack {attack_name}_temporal_propagation \\")
    print(f"    --folder {folder_name}/adversarial_sequence \\")
    print(f"    --num_samples {T}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Temporal propagation adversarial attack')
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
                        help='Temporal window size to perturb (default: 5 frames)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"TEMPORAL PROPAGATION ATTACK - {args.attack}")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'Trained', 'network.pt')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    # Create output directory
    output_base = os.path.join(os.path.dirname(__file__), f'{args.attack}_temporal_propagation')
    os.makedirs(output_base, exist_ok=True)
    
    # Create folder name
    eps_str = f"{args.eps}".rstrip('0').rstrip('.').replace('.', '')
    alpha_str = f"{args.alpha}".rstrip('0').rstrip('.').replace('.', '')
    decay_str = f"{args.decay}".rstrip('0').rstrip('.').replace('.', '')
    
    if args.attack == 'FGSM':
        folder_name = f"results_eps{eps_str}_win{args.window_size}"
    elif args.attack == 'PGD':
        folder_name = f"results_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_win{args.window_size}"
        if args.random_start:
            folder_name += "_randstart"
    elif args.attack == 'MIFGSM':
        folder_name = f"results_eps{eps_str}_alpha{alpha_str}_steps{args.steps}_decay{decay_str}_win{args.window_size}"
    
    output_dir = os.path.join(output_base, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load PyTorch model (for adversarial generation)
    print("Loading PyTorch model (for attack generation)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    pytorch_model = Network().to(device)
    current_model_dict = pytorch_model.state_dict()
    loaded_state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }
    pytorch_model.load_state_dict(new_state_dict, strict=False)
    pytorch_model.train()
    print("PyTorch model loaded\n")
    
    # Load Lava SDNN (for inference/evaluation)
    print("Loading Lava SDNN (for inference/evaluation)")
    pilotnet_dir = os.path.join(base_dir, 'pilotnet_sdnn')
    os.chdir(pilotnet_dir)
    lava_net = netx.hdf5.Network(net_config='network.net', skip_layers=1)
    net_config = lava_net.net_config
    os.chdir(base_dir)
    print("Lava SDNN loaded\n")
    
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
    
    for img_file in image_files:
        img_path = os.path.join(testing_dataset_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform_to_tensor(img)
        tensor_list.append(img_tensor)
    
    if len(tensor_list) == 0:
        print("No images found in testing_dataset")
        return
    
    print(f"Loaded {len(tensor_list)} images\n")
    
    # Stack into sequence tensor [1, C, H, W, T]
    sequence_tensor = torch.stack(tensor_list, dim=3).unsqueeze(0).to(device)
    print(f"Sequence tensor shape: {sequence_tensor.shape}\n")
    
    # Load ground truth labels and original predictions
    print("Loading ground truth labels and original predictions")
    results_path = os.path.join(base_dir, 'pilotnet_sdnn', 'results.txt')
    ground_truth_labels, original_predictions = load_ground_truth_and_predictions(results_path)
    print(f"Loaded ground truth and predictions for {len(ground_truth_labels)} images\n")
    
    # Prepare attack parameters
    attack_params = {
        'attack': args.attack,
        'eps': args.eps,
        'alpha': args.alpha,
        'steps': args.steps,
        'decay': args.decay,
        'random_start': args.random_start
    }
    
    # Run temporal propagation attack
    results = temporal_propagation_attack(
        pytorch_model, lava_net, net_config, sequence_tensor, attack_params, 
        ground_truth_labels, original_predictions, output_dir, folder_name, window_size=args.window_size
    )
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'temporal_propagation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Temporal Propagation Attack Results (NON-OVERLAPPING WINDOWS)\n")
        f.write(f"{'='*120}\n")
        f.write(f"Attack: {args.attack}\n")
        f.write(f"Window size (perturbed): {args.window_size}\n")
        f.write(f"Frames per window: {args.window_size + 1} ({args.window_size} adversarial + 1 clean)\n")
        f.write(f"Epsilon: {args.eps}\n")
        f.write(f"Number of windows tested: {len(results)}\n")
        f.write(f"Pattern: [ADV, ADV, ..., ADV, CLEAN] repeated\n")
        f.write(f"Inference: Lava SDNN (same as deployment)\n\n")
        f.write(f"Window | Adversarial Frames | Clean Frame | Ground Truth  | Lava Original | Lava Mixed Seq | Diff      | Orig Error | Adv Error | Pert Norm\n")
        f.write(f"{'-'*140}\n")
        for r in results:
            f.write(f"{r['window_idx']:6d} | ")
            f.write(f"[{r['perturbed_frames'][0]:3d}, {r['perturbed_frames'][1]:3d}]       | ")
            f.write(f"{r['clean_frame']:11d} | ")
            f.write(f"{r['ground_truth']:13.6f} | ")
            f.write(f"{r['lava_original_prediction']:13.6f} | ")
            f.write(f"{r['lava_mixed_prediction']:14.6f} | ")
            f.write(f"{r['prediction_difference']:9.6f} | ")
            f.write(f"{r['original_error']:10.6f} | ")
            f.write(f"{r['adversarial_error']:9.6f} | ")
            f.write(f"{r['perturbation_norm']:9.6f}\n")
    
    print(f"\n{'='*70}")
    print(f"Results saved to {results_file}\n")
    
    # Calculate statistics
    prediction_diffs = [r['prediction_difference'] for r in results]
    mean_diff = np.mean(prediction_diffs)
    max_diff = np.max(prediction_diffs)
    std_diff = np.std(prediction_diffs)
    
    original_errors = [r['original_error'] for r in results]
    adversarial_errors = [r['adversarial_error'] for r in results]
    mean_orig_error = np.mean(original_errors)
    mean_adv_error = np.mean(adversarial_errors)
    
    print(f"Statistical Summary:")
    print(f"{'='*70}")
    print(f"Mean prediction difference (Orig vs Adv): {mean_diff:.6f}")
    print(f"Max prediction difference:                 {max_diff:.6f}")
    print(f"Std prediction difference:                 {std_diff:.6f}")
    print(f"Mean original error (vs GT):               {mean_orig_error:.6f}")
    print(f"Mean adversarial error (vs GT):            {mean_adv_error:.6f}")
    print(f"Error increase:                            {mean_adv_error - mean_orig_error:.6f}\n")
    
    # Create visualization plot
    print("Creating visualization plots")
    
    # Single plot: Prediction comparison over frames
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    window_indices = [r['window_idx'] for r in results]
    ground_truths = [r['ground_truth'] for r in results]
    lava_original_preds = [r['lava_original_prediction'] for r in results]
    lava_mixed_preds = [r['lava_mixed_prediction'] for r in results]
    clean_frame_indices = [r['clean_frame'] for r in results]
    
    # Plot predictions - THREE LINES COMPARISON (All Lava SDNN)
    ax.plot(clean_frame_indices, ground_truths, label='Ground Truth (What it should be)', 
            linewidth=2.5, marker='s', markersize=3, alpha=0.9, color='green')
    ax.plot(clean_frame_indices, lava_original_preds, label='Lava Original Prediction (All frames clean)', 
            linewidth=2, marker='o', markersize=3, alpha=0.8, color='blue', linestyle='--')
    ax.plot(clean_frame_indices, lava_mixed_preds, label='Lava Mixed Sequence Prediction (With adv history)', 
            linewidth=2, marker='x', markersize=3, alpha=0.9, color='red')
    ax.set_xlabel('Clean Frame Index', fontsize=11)
    ax.set_ylabel('Steering Angle (radians)', fontsize=11)
    ax.set_title(f'Temporal Propagation Attack - {args.attack} (Window Size={args.window_size}, Non-Overlapping)\nLava SDNN Inference: How Adversarial History Affects Clean Frame Predictions', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'temporal_propagation_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {plot_path}\n")
    
    print(f"{'='*70}")
    print(f"Temporal Propagation Attack Complete!")
    print(f"{'='*70}")
    print(f"Procedure: SOUND (All predictions use Lava SDNN)")
    print(f"  - Original sequence: Lava SDNN inference")
    print(f"  - Adversarial generation: PyTorch (gradient-based)")
    print(f"  - Mixed sequence evaluation: Lava SDNN inference")
    print(f"  - Result: Attack effect measured using deployment model")
    print(f"{'='*70}")
    print(f"Adversarial images saved in: {os.path.join(output_dir, 'adversarial_windows')}/")
    print(f"Verification images saved in: {os.path.join(output_dir, 'verification')}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
