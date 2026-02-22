"""
Quick script to verify that adversarial images are actually perturbed
"""

import os
import argparse
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Verify adversarial perturbations')
    parser.add_argument('--attack', type=str, required=True,
                        help='Attack directory (e.g., PGD, FGSM)')
    parser.add_argument('--folder', type=str, required=True,
                        help='Adversarial images folder (e.g., adv_img_eps003_alpha0007_steps10)')
    parser.add_argument('--torchattacks', action='store_true',
                        help='Check Using_torchattacks directory instead of main directory')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Verifying Adversarial Perturbations")
    print(f"{'='*70}\n")
    
    # Paths
    base_dir = os.path.dirname(__file__)
    testing_dir = os.path.join(base_dir, 'testing_dataset')
    
    if args.torchattacks:
        adv_dir = os.path.join(base_dir, 'Using_torchattacks', args.attack, args.folder)
        print(f"Checking: Using_torchattacks/{args.attack}/{args.folder}/\n")
    else:
        adv_dir = os.path.join(base_dir, args.attack, args.folder)
        print(f"Checking: {args.attack}/{args.folder}/\n")
    
    if not os.path.exists(adv_dir):
        print(f"✗ Error: Directory not found: {adv_dir}")
        return
    
    # Get image files
    clean_files = sorted([f for f in os.listdir(testing_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('.')[0]))
    adv_files = sorted([f for f in os.listdir(adv_dir) if f.endswith('.jpg')],
                      key=lambda x: int(x.split('.')[0]))
    
    print(f"Clean images: {len(clean_files)}")
    print(f"Adversarial images: {len(adv_files)}\n")
    
    # Check first 10 images
    num_to_check = min(10, len(clean_files), len(adv_files))
    
    print(f"Checking first {num_to_check} images...\n")
    print(f"{'Image':<15}{'Mean Diff':<15}{'Max Diff':<15}{'Perturbed?':<15}")
    print("-"*60)
    
    perturbed_count = 0
    
    for i in range(num_to_check):
        clean_path = os.path.join(testing_dir, clean_files[i])
        adv_path = os.path.join(adv_dir, adv_files[i])
        
        # Load images
        clean_img = np.array(Image.open(clean_path))
        adv_img = np.array(Image.open(adv_path))
        
        # Calculate difference
        diff = np.abs(clean_img.astype(float) - adv_img.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        is_perturbed = mean_diff > 0.1  # Threshold
        if is_perturbed:
            perturbed_count += 1
        
        status = "YES ✓" if is_perturbed else "NO ✗"
        print(f"{clean_files[i]:<15}{mean_diff:<15.4f}{max_diff:<15.4f}{status:<15}")
    
    print("-"*60)
    print(f"\nSummary: {perturbed_count}/{num_to_check} images are perturbed\n")
    
    if perturbed_count == 0:
        print("⚠ WARNING: No perturbations detected!")
        print("  Check if the attack generation script is working correctly.")
    elif perturbed_count < num_to_check:
        print(f"⚠ WARNING: Only {perturbed_count}/{num_to_check} images are perturbed!")
    else:
        print("✓ All checked images have perturbations.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
