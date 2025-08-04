import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import lava.lib.dl.slayer as slayer

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


#modular class attack

class Attack:
    """
    Base class for all attacks.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class PGD(Attack):
    """
    PGD attack for a regression model.
    """
    def __init__(self, model, eps=0.03, alpha=0.007, steps=10):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images, target):
        images = images.clone().detach().to(self.device)
        target = target.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Get model output for the current adversarial image
            outputs, _, _ = self.model(adv_images)
            final_prediction = outputs.mean()
            
            # Calculate loss (MSE for regression)
            loss = nn.MSELoss()(final_prediction, target.squeeze())
            
            # Get gradient of loss w.r.t. the input
            grad = torch.autograd.grad(loss, adv_images,
            retain_graph=False, create_graph=False)[0]

            # Perform the PGD step
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images



if __name__ == "__main__":

    #__file__ --> this code, os.path.dirname(__file__) --> PGD folder, os.path.dirname(os.path.dirname(__file__)) --> AdvAttk folder

    path_to_frames = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testing_dataset')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Trained', 'network.pt')  
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results.txt')  

    sequence_tensor = None
    try:

        all_files = os.listdir(path_to_frames)
        frame_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Sorting files based on the number in the filename
        frame_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
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

    # PGD attack
    attack = PGD(model, eps=0.03, alpha=0.007, steps=10)
    target_label = torch.tensor([[5.0]]) # Example target

    original_input = sequence_tensor.unsqueeze(0)
    adversarial_input = attack(original_input, target_label)

    print("Adversarial input generated via PGD.")

    # final pred
    with torch.no_grad():
        original_pred_tensor, _, _ = model(original_input)
        adversarial_pred_tensor, _, _ = model(adversarial_input)

    #Comparison Plot
    ground_truth_angles = [float(row[1]) for row in original_results_data]  # Get ground truth (2nd column) from each row
    time_steps = np.arange(len(ground_truth_angles))
    original_pred_np = original_pred_tensor.squeeze().cpu().numpy()
    adversarial_pred_np = adversarial_pred_tensor.squeeze().cpu().numpy()

    min_len = min(len(time_steps), len(original_pred_np), len(adversarial_pred_np))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps[:min_len], ground_truth_angles[:min_len], label='Ground Truth', color='g', linewidth=2)
    plt.plot(time_steps[:min_len], original_pred_np[:min_len], label='Original Prediction', color='b', linestyle='--')
    plt.plot(time_steps[:min_len], adversarial_pred_np[:min_len], label='Adversarial Prediction', color='r', linestyle='-.')
    
    plt.title('Effect of PGD Attack on Steering Angle Prediction')
    plt.xlabel('Time Step (Frame)')
    plt.ylabel('Steering Angle')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(os.path.dirname(__file__), 'Results', 'attack_comparison.png')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    plt.savefig(plot_filename)
    print(f"\n Comparison plot saved as '{plot_filename}'")

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