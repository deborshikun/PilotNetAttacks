import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import lava.lib.dl.slayer as slayer

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  
from Attacks.attacks import MIFGSM  

# A placeholder for the loss function 
def event_rate_loss(x):
    return torch.tensor(0.0)

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        sdnn_params = {
            'threshold'   : 0.1, 'tau_grad'    : 0.5, 'scale_grad'  : 1,
            'requires_grad' : True, 'shared_param'  : True, 'activation'  : F.relu,
        }
        sdnn_cnn_params = {
            **sdnn_params, 'norm' : slayer.neuron.norm.MeanOnlyBatchNorm,
        }
        sdnn_dense_params = {
            **sdnn_cnn_params, 'dropout' : slayer.neuron.Dropout(p=0.2),
        }
        self.blocks = nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params,  3, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 64*40, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   100,  50, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,    50,  10, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Output(sdnn_dense_params,   10,   1, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x):
        count = []
        event_cost = 0
        for block in self.blocks:
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum((x[..., 1:] > 0).to(x.dtype)).item())
        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

if __name__ == "__main__":

    #__file__ --> this code, os.path.dirname(__file__) --> PGD folder, os.path.dirname(os.path.dirname(__file__)) --> AdvAttk folder

    path_to_frames = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testing_dataset')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Trained', 'network.pt')  
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results.txt')  

    sequence_tensor = None
    try:
        frame_files = sorted(os.listdir(path_to_frames))
        transform = transforms.Compose([
            transforms.Resize((33, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor_list = [transform(Image.open(os.path.join(path_to_frames, f)).convert('RGB')) for f in frame_files]
        if tensor_list:
            sequence_tensor = torch.stack(tensor_list, dim=3)
            print(f"Loaded {len(frame_files)} frames. Final tensor shape: {sequence_tensor.shape}")
    except FileNotFoundError:
        print(f"Error: The directory '{path_to_frames}' was not found.")
        exit()

    # Load Original Results Data

    original_results_data = []
    try:
        with open(results_path, 'r') as f:
            next(f) # Skip header 
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    # Store [ImageName, GroundTruth, OriginalPrediction]
                    original_results_data.append(parts[0:3])
        print(f"Loaded {len(original_results_data)} lines from original results file.")
    except FileNotFoundError:
        print(f"Error: Ground truth file '{results_path}' not found.")
        exit()

    # Load Model
    if sequence_tensor is not None:    
        model = Network()
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        }
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print(" Model 'network.pt' loaded successfully.")
        # net = Network().to('cpu')
        # current_model_dict = net.state_dict()
        # loaded_state_dict = torch.load(model_path, map_location='cpu')
        # new_state_dict = {
        # k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        # for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        # }
        # net.load_state_dict(new_state_dict, strict=False)

    # FGSM Attack
    attack = MIFGSM(model, eps=0.03, alpha=2/255, steps=10, decay=1.0)
    target_label = torch.tensor([[5.0]]) # Example target to drive the loss
    
    original_input = sequence_tensor.unsqueeze(0)
    adversarial_input = attack(original_input, target_label)
    
    print("Adversarial input generated via PGD.")

    # Comparison Text
    with torch.no_grad():
        adversarial_pred_tensor, _, _ = model(adversarial_input)
    
    adversarial_pred_list = adversarial_pred_tensor.squeeze().cpu().numpy().tolist()
    
    output_filename = os.path.join(os.path.dirname(__file__), 'Results', 'adversarial_comparison.txt')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w') as f:
        f.write("ImageName\tGroundTruth\tOriginalPrediction\tAdversarialPrediction\n")
        num_lines = min(len(original_results_data), len(adversarial_pred_list))
        for i in range(num_lines):
            image_name, ground_truth, original_pred = original_results_data[i]
            adversarial_pred = adversarial_pred_list[i]
            f.write(f"{image_name}\t{ground_truth}\t{original_pred}\t{adversarial_pred:.8f}\n")
            
    print(f"\n Comparison results saved to '{output_filename}'")

    
    # Save adversarial images
    save_dir = os.path.join(os.path.dirname(__file__), 'Results', 'adv_images')
    os.makedirs(save_dir, exist_ok=True)
    # Remove batch dimension and denormalize
    adv_imgs = adversarial_input.squeeze(0)
    for idx in range(adv_imgs.shape[3]):
        img = adv_imgs[..., idx]
         # Denormalize
        img = img * 0.5 + 0.5
        img = img.clamp(0, 1)
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(os.path.join(save_dir, f"adv_{idx:04d}.png"))
    print(f"Adversarial images saved to '{save_dir}'.")