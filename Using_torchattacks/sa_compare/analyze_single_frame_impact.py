"""
Analyze Single Frame Impact - Growing Window Approach

This script analyzes pre-generated adversarial frames to determine the incremental
impact of perturbing each frame as the temporal window grows.

REQUIRES: Pre-generated adversarial frames from generate_single_frame_adversarial.py

Uses LAVA SDNN inference (NOT PyTorch) for accurate, consistent predictions.

For each iteration i (1 to 200):
1. Run Lava inference on clean frames [0 to i]
2. Run Lava inference on frames [0 to i-1 clean, i perturbed]
3. Compare the two steering angles

This reveals the incremental impact of each frame as it's added to the sequence.
"""

import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add pilotnet_sdnn to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pilotnet_sdnn'))

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.lib.dl import netx
from dataset import PilotNetDataset
from utils import PilotNetEncoder, PilotNetDecoder, CustomSimRunConfig, get_input_transform


def run_lava_inference(temp_data_dir, net, transform, num_samples):
    """
    Run Lava SDNN inference on images in temp_data/driving_dataset/.
    
    Args:
        temp_data_dir: Path to temp_data directory (contains driving_dataset/)
        net: Loaded Lava network
        transform: Image transformation function
        num_samples: Number of samples to process
    
    Returns:
        Final steering angle prediction (float)
    """
    # Setup dataset (temp_data_dir should contain driving_dataset/ with images and data.txt)
    dataset = PilotNetDataset(
        path=temp_data_dir,
        size=net.inp.shape[:2],
        transform=transform,
        visualize=True,
        sample_offset=0
    )
    
    # Setup Lava processes
    num_steps = num_samples + len(net.layers)
    out_offset = len(net.layers) + 3
    
    dataloader = io.dataloader.SpikeDataloader(dataset=dataset)
    input_encoder = PilotNetEncoder(
        shape=net.inp.shape,
        net_config=net.net_config,
        compression=io.encoder.Compression.DENSE
    )
    output_decoder = PilotNetDecoder(shape=net.out.shape)
    
    gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
    output_logger = io.sink.RingBuffer(shape=net.out_layer.shape, buffer=num_steps)
    
    # Connect processes
    dataloader.ground_truth.connect(gt_logger.a_in)
    dataloader.s_out.connect(input_encoder.inp)
    input_encoder.out.connect(net.inp)
    net.out.connect(output_decoder.inp)
    output_decoder.out.connect(output_logger.a_in)
    
    # Run inference
    run_config = CustomSimRunConfig()
    net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
    
    output = output_logger.data.get().flatten()
    net.stop()
    
    # Return the final prediction (last valid output after pipeline delay)
    if len(output) > out_offset + num_samples - 1:
        return output[out_offset + num_samples - 1]
    else:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Analyze single frame impact using pre-generated adversarial frames')
    parser.add_argument('--attack', type=str, required=True,
                        help='Attack name (FGSM, PGD, MIFGSM)')
    parser.add_argument('--folder', type=str, required=True,
                        help='Folder name containing adversarial images (e.g., single_frame_eps003_alpha0007_steps10)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Single Frame Impact Analysis - {args.attack}")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pilotnet_dir = os.path.join(base_dir, 'pilotnet_sdnn')
    testing_dataset_dir = os.path.join(base_dir, 'testing_dataset')
    
    attack_dir = os.path.join(os.path.dirname(__file__), args.attack)
    adv_images_dir = os.path.join(attack_dir, f'adv_images_{args.folder}')
    checking_dir = os.path.join(attack_dir, f'checking_{args.folder}')
    results_path = os.path.join(attack_dir, f'results_{args.folder}.txt')
    csv_path = os.path.join(attack_dir, f'results_{args.folder}.csv')
    
    # Check if adversarial images exist
    if not os.path.exists(adv_images_dir):
        print(f"ERROR: Adversarial images not found at {adv_images_dir}")
        print(f"\nPlease generate adversarial frames first using:")
        print(f"  py -3.9 Using_torchattacks/sa_compare/generate_single_frame_adversarial.py \\")
        print(f"    --attack {args.attack} [--eps EPS] [--alpha ALPHA] [--steps STEPS]")
        return
    
    os.makedirs(checking_dir, exist_ok=True)
    
    print(f"Loading adversarial frames from: {adv_images_dir}\n")
    
    # Load Lava SDNN network
    print("Loading Lava SDNN network...")
    os.chdir(pilotnet_dir)
    net = netx.hdf5.Network(net_config='network.net', skip_layers=1)
    transform = get_input_transform(net.net_config)
    print("  Network loaded\n")
    
    # Get list of clean and adversarial images
    print("Loading image lists...")
    clean_image_files = sorted([f for f in os.listdir(testing_dataset_dir) if f.endswith('.jpg')],
                                key=lambda x: int(x.split('.')[0]))
    adv_image_files = sorted([f for f in os.listdir(adv_images_dir) if f.endswith('.jpg')],
                              key=lambda x: int(x.split('.')[0]))
    
    print(f"  Clean frames: {len(clean_image_files)}")
    print(f"  Adversarial frames: {len(adv_image_files)}\n")
    
    if len(clean_image_files) != len(adv_image_files):
        print(f"WARNING: Mismatch in frame count")
        num_frames = min(len(clean_image_files), len(adv_image_files))
        print(f"  Using first {num_frames} frames\n")
    else:
        num_frames = len(clean_image_files)
    
    # Growing window analysis
    print(f"Running growing window analysis with Lava SDNN...")
    print(f"  Iteration 1: [f0 clean] vs [f0 perturbed]")
    print(f"  Iteration 2: [f0, f1 clean] vs [f0 clean, f1 perturbed]")
    print(f"  Iteration 3: [f0, f1, f2 clean] vs [f0, f1 clean, f2 perturbed]")
    print(f"  ... and so on up to {num_frames} frames")
    print(f"  Total Lava inferences: {num_frames * 2}\n")
    
    results = []
    
    for i in tqdm(range(num_frames), desc="Analyzing frames"):
        window_size = i + 1
        
        # ===== 1. CLEAN SEQUENCE: [f0, f1, ..., fi] =====
        temp_clean_dir = os.path.join(attack_dir, 'temp_clean')
        clean_dataset_dir = os.path.join(temp_clean_dir, 'driving_dataset')
        os.makedirs(clean_dataset_dir, exist_ok=True)
        
        # Copy clean images [0...i]
        for idx in range(window_size):
            shutil.copy(
                os.path.join(testing_dataset_dir, clean_image_files[idx]),
                os.path.join(clean_dataset_dir, clean_image_files[idx])
            )
        
        # Create data.txt for clean sequence
        # Pad with duplicates to ensure dataset is large enough for dataloader (needs ~200+ entries)
        with open(os.path.join(clean_dataset_dir, 'data.txt'), 'w') as f:
            # Write original entries
            for idx in range(window_size):
                f.write(f"{clean_image_files[idx]} 0.0\n")
            # Pad by repeating the sequence until we have 200+ entries
            entries_written = window_size
            while entries_written < 200:
                for idx in range(window_size):
                    f.write(f"{clean_image_files[idx]} 0.0\n")
                    entries_written += 1
                    if entries_written >= 200:
                        break
        
        # Run inference on clean sequence
        clean_sa = run_lava_inference(temp_clean_dir, net, transform, window_size)
        
        # Cleanup clean temp directory
        shutil.rmtree(temp_clean_dir, ignore_errors=True)
        
        # ===== 2. PERTURBED SEQUENCE: [f0...f(i-1) clean, fi perturbed] =====
        temp_pert_dir = os.path.join(attack_dir, 'temp_pert')
        pert_dataset_dir = os.path.join(temp_pert_dir, 'driving_dataset')
        os.makedirs(pert_dataset_dir, exist_ok=True)
        
        # Copy clean frames [0...i-1]
        for idx in range(i):
            shutil.copy(
                os.path.join(testing_dataset_dir, clean_image_files[idx]),
                os.path.join(pert_dataset_dir, clean_image_files[idx])
            )
        
        # Copy adversarial frame i
        shutil.copy(
            os.path.join(adv_images_dir, adv_image_files[i]),
            os.path.join(pert_dataset_dir, adv_image_files[i])
        )
        
        # Create data.txt for perturbed sequence
        # Pad with duplicates to ensure dataset is large enough for dataloader (needs ~200+ entries)
        with open(os.path.join(pert_dataset_dir, 'data.txt'), 'w') as f:
            # Write original entries
            for idx in range(window_size):
                if idx < i:
                    f.write(f"{clean_image_files[idx]} 0.0\n")
                else:
                    f.write(f"{adv_image_files[idx]} 0.0\n")
            # Pad by repeating the sequence until we have 200+ entries
            entries_written = window_size
            while entries_written < 200:
                for idx in range(window_size):
                    if idx < i:
                        f.write(f"{clean_image_files[idx]} 0.0\n")
                    else:
                        f.write(f"{adv_image_files[idx]} 0.0\n")
                    entries_written += 1
                    if entries_written >= 200:
                        break
        
        # Run inference on perturbed sequence
        perturbed_sa = run_lava_inference(temp_pert_dir, net, transform, window_size)
        
        # Cleanup perturbed temp directory
        shutil.rmtree(temp_pert_dir, ignore_errors=True)
        
        # Calculate impact
        absolute_diff = perturbed_sa - clean_sa
        percent_diff = (absolute_diff / clean_sa * 100) if clean_sa != 0 else 0.0
        
        # Store results
        results.append({
            'iteration': i + 1,
            'frame': i,
            'image': clean_image_files[i],
            'window_size': window_size,
            'clean_sa': clean_sa,
            'perturbed_sa': perturbed_sa,
            'absolute_diff': absolute_diff,
            'percent_diff': percent_diff
        })
        
        # Create verification images (first 20, last 10, every 10th)
        save_image = (i < 20) or (i >= num_frames - 10) or (i % 10 == 0)
        
        if save_image:
            fig, axes = plt.subplots(3, window_size, figsize=(3*window_size, 9))
            if window_size == 1:
                axes = axes.reshape(3, 1)
            
            # Top row: Clean window
            for j in range(window_size):
                img_path = os.path.join(testing_dataset_dir, clean_image_files[j])
                img = Image.open(img_path).convert('RGB')
                axes[0, j].imshow(img)
                axes[0, j].set_title(f'Clean f{j}', fontsize=8)
                axes[0, j].axis('off')
            
            # Middle row: Perturbed window
            for j in range(window_size):
                if j == i:
                    img_path = os.path.join(adv_images_dir, adv_image_files[j])
                    img = Image.open(img_path).convert('RGB')
                    axes[1, j].imshow(img)
                    axes[1, j].set_title(f'PERT f{j}', fontsize=8, color='red', weight='bold')
                else:
                    img_path = os.path.join(testing_dataset_dir, clean_image_files[j])
                    img = Image.open(img_path).convert('RGB')
                    axes[1, j].imshow(img)
                    axes[1, j].set_title(f'Clean f{j}', fontsize=8)
                axes[1, j].axis('off')
            
            # Bottom row: Perturbation
            for j in range(window_size):
                if j == i:
                    clean_img = np.array(Image.open(os.path.join(testing_dataset_dir, clean_image_files[j])))
                    adv_img = np.array(Image.open(os.path.join(adv_images_dir, adv_image_files[j])))
                    pert = np.abs(adv_img.astype(float) - clean_img.astype(float)).mean(axis=2)
                    p_max = pert.max()
                    axes[2, j].imshow(pert, cmap='hot')
                    axes[2, j].set_title(f'Pert (max={p_max:.1f})', fontsize=8, color='red', weight='bold')
                axes[2, j].axis('off')
            
            fig.suptitle(f'Iteration {i+1} (Window Size {window_size})\n'
                        f'Clean SA: {clean_sa:.4f} | Perturbed SA: {perturbed_sa:.4f} | Diff: {absolute_diff:.4f}',
                        fontsize=10, weight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(checking_dir, f'w{i+1}.jpg'), dpi=100, bbox_inches='tight')
            plt.close()
    
    print(f"\nAnalysis complete!\n")
    
    # Save results
    os.chdir(base_dir)
    print(f"Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Single Frame Perturbation Analysis - Growing Window - {args.attack}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Adversarial Frames: {adv_images_dir}\n")
        f.write(f"Approach: Growing Window (Lava SDNN Inference)\n")
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
        for idx, r in enumerate(sorted_by_impact[:10], 1):
            f.write(f"  {idx}. Iteration {r['iteration']} (Frame {r['frame']}, {r['image']}, Window={r['window_size']}): "
                   f"{abs(r['absolute_diff']):.6f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    # Save CSV
    with open(csv_path, 'w') as f:
        f.write("Iteration,Frame,WindowSize,Image,CleanSA,PerturbedSA,AbsDiff,PercentDiff\n")
        for r in results:
            f.write(f"{r['iteration']},{r['frame']},{r['window_size']},{r['image']},"
                   f"{r['clean_sa']:.6f},{r['perturbed_sa']:.6f},"
                   f"{r['absolute_diff']:.6f},{r['percent_diff']:.2f}\n")
    
    print(f"  Results saved!\n")
    
    # Print summary
    abs_diffs = [abs(r['absolute_diff']) for r in results]
    sorted_by_impact = sorted(results, key=lambda x: abs(x['absolute_diff']), reverse=True)
    print(f"{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(f"Total frames analyzed:       {num_frames}")
    print(f"Total Lava inferences:       {num_frames * 2}")
    print(f"Mean absolute impact:        {np.mean(abs_diffs):.6f}")
    print(f"Max absolute impact:         {np.max(abs_diffs):.6f} (Iteration {sorted_by_impact[0]['iteration']}, Frame {sorted_by_impact[0]['frame']})")
    print(f"Min absolute impact:         {np.min(abs_diffs):.6f} (Iteration {sorted_by_impact[-1]['iteration']}, Frame {sorted_by_impact[-1]['frame']})")
    print(f"\nVerification images:         {checking_dir}/")
    print(f"Detailed results saved to:   {results_path}")
    print(f"CSV file (for Excel):        {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

