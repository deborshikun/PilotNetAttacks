"""
Run Inference on Adversarial Images using lava-dl PilotNet SDNN using the network.net model

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add pilotnet_sdnn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pilotnet_sdnn'))

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.lib.dl import netx
from dataset import PilotNetDataset
from utils import PilotNetEncoder, PilotNetDecoder, CustomSimRunConfig, get_input_transform


def load_original_results(results_path):
    """Load original inference results"""
    results = []
    with open(results_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                results.append({
                    'image': parts[0],
                    'gt': float(parts[1]),
                    'original': float(parts[2])
                })
    return results


def create_data_txt_for_adversarial(adv_images_dir, original_results, output_path):
    """Create data.txt file for adversarial dataset"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get list of adversarial images
    adv_files = sorted([f for f in os.listdir(adv_images_dir) if f.endswith(('.jpg', '.png'))],
                       key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    print(f"  Found {len(adv_files)} adversarial images")
    print(f"  Original results contains {len(original_results)} entries")
    
    # Map from adversarial image numbers to original image names
    # testing_dataset has 0.jpg-200.jpg, but original_results has 10550.jpg etc
    # We need to map by index
    with open(output_path, 'w') as f:
        for idx, adv_file in enumerate(adv_files):
            if idx < len(original_results):
                gt_radians = original_results[idx]['gt']
                gt_degrees = gt_radians * 180 / np.pi
                f.write(f"{adv_file} {gt_degrees}\n")
    
    print(f"  Created data.txt with {min(len(adv_files), len(original_results))} entries")


def main():
    parser = argparse.ArgumentParser(description='Run inference on adversarial images')
    parser.add_argument('--attack', type=str, required=True,
                        help='Attack name (FGSM, PGD, MIFGSM)')
    parser.add_argument('--folder', type=str, required=True,
                        help='Folder name containing adversarial images (e.g., adv_img_eps0.03)')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to process')
    
    args = parser.parse_args()
    
    print(f"Inference on Adversarial Images - {args.attack}")
    print(f"{'-'*45}\n")
    
    # Paths
    base_dir = os.path.dirname(__file__)
    pilotnet_dir = os.path.join(base_dir, 'pilotnet_sdnn')
    attack_dir = os.path.join(base_dir, args.attack)
    adv_images_dir = os.path.join(attack_dir, args.folder)
    original_results_path = os.path.join(pilotnet_dir, 'results.txt')
    output_results_path = os.path.join(attack_dir, f'results_{args.folder}.txt')
    output_plot_path = os.path.join(attack_dir, f'comparison_{args.folder}.png')
    
    # Check if adversarial images exist
    if not os.path.exists(adv_images_dir):
        print(f"✗ Error: Adversarial images not found at {adv_images_dir}")
        print(f"  Make sure you generated adversarial images first.")
        return
    
    # Load original results
    print("Loading original results")
    original_results = load_original_results(original_results_path)
    print(f"  Loaded {len(original_results)} original results\n")
    
    # Create temporary data structure for adversarial images
    print("Preparing adversarial dataset")
    temp_data_dir = os.path.join(attack_dir, 'temp_data', 'driving_dataset')
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Create data.txt
    data_txt_path = os.path.join(temp_data_dir, 'data.txt')
    create_data_txt_for_adversarial(adv_images_dir, original_results, data_txt_path)
    
    # Copy adversarial images to temp_data #Deborshi 
    #needed for PilotNetDataset
    import shutil
    for img_file in os.listdir(adv_images_dir):
        if img_file.endswith(('.jpg', '.png')):
            shutil.copy(
                os.path.join(adv_images_dir, img_file),
                os.path.join(temp_data_dir, img_file)
            )
    print(f"  Prepared dataset\n")
    
    # Load network
    print("Loading network")
    os.chdir(pilotnet_dir)  # Change to pilotnet_sdnn directory
    net = netx.hdf5.Network(net_config='network.net', skip_layers=1)
    print(f"  Network loaded\n")
    
    # Setup dataset
    print("Setting up dataset")
    transform = get_input_transform(net.net_config)
    temp_data_parent = os.path.join(attack_dir, 'temp_data')
    
    adv_dataset = PilotNetDataset(
        path=temp_data_parent,
        size=net.inp.shape[:2],
        transform=transform,
        visualize=True,
        #sample_offset=0
        sample_offset=10550  # To match original indexing
    )
    print(f"  Dataset ready with {len(adv_dataset)} samples\n")
    
    # Setup Lava processes
    print("Setting up Lava processes")
    num_samples = min(args.num_samples, len(adv_dataset))
    num_steps = num_samples + len(net.layers)
    out_offset = len(net.layers) + 3
    
    dataloader = io.dataloader.SpikeDataloader(dataset=adv_dataset)
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
    print(f"Processes connected\n")
    
    # Run inference
    print(f"Running inference on {num_samples} adversarial images")
    run_config = CustomSimRunConfig()
    net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
    
    output = output_logger.data.get().flatten()
    gts = gt_logger.data.get().flatten()
    net.stop()
    print(f"Inference complete\n")
    
    # Save results
    print("Saving results")
    os.chdir(base_dir)  # Change back to base directory
    
    max_outputs = len(output) - out_offset
    num_to_save = min(num_samples, max_outputs, len(original_results))
    
    with open(output_results_path, 'w') as f:
        f.write("ImageName\t\tGroundTruth\t\tOriginalOutput\t\tAdversarialOutput\n")
        for idx in range(num_to_save):
            img_name = original_results[idx]['image']
            gt = original_results[idx]['gt']
            original_out = original_results[idx]['original']
            adv_out = output[out_offset + idx]
            f.write(f"{img_name}\t{gt}\t\t\t{original_out}\t\t\t{adv_out}\n")
    
    print(f"Results saved to {output_results_path}\n")
    
    # Generate plot
    print("Generating comparison plot")
    gts_plot = [original_results[i]['gt'] for i in range(num_to_save)]
    originals_plot = [original_results[i]['original'] for i in range(num_to_save)]
    adversarial_plot = [output[out_offset + i] for i in range(num_to_save)]
    
    # Extract image numbers for x-axis
    image_numbers = [int(original_results[i]['image'].replace('.jpg', '')) for i in range(num_to_save)]
    
    plt.figure(figsize=(7, 5))
    plt.plot(image_numbers, gts_plot, label='Ground Truth', linewidth=2)
    plt.plot(image_numbers, originals_plot, label='Original Prediction', linewidth=2)
    plt.plot(image_numbers, adversarial_plot, label=f'{args.attack} Output', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Steering angle (radians)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    print(f"Plot saved to {output_plot_path}\n")
    
    # Calculate statistics
    # mse_original = np.mean([(gts[i] - originals_plot[i])**2 for i in range(num_to_save)])
    # mse_adversarial = np.mean([(gts[i] - adversarial_plot[i])**2 for i in range(num_to_save)])
    
    # print(f"{'='*60}")
    # print("Results Summary:")
    # print(f"{'='*60}")
    # print(f"MSE (Original):      {mse_original:.6f}")
    # print(f"MSE (Adversarial):   {mse_adversarial:.6f}")
    # print(f"MSE Increase:        {((mse_adversarial - mse_original) / mse_original * 100):.2f}%")
    # print(f"\n  Inference pipeline completed!")
    # print(f"{'='*60}\n")
    
    # Cleanup temp directory
    shutil.rmtree(os.path.join(attack_dir, 'temp_data'))


if __name__ == "__main__":
    main()
