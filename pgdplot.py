import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Assuming lava.lib.dl.slayer is installed
import lava.lib.dl.slayer as slayer

# A placeholder for the loss function needed by the model definition.
def event_rate_loss(x):
    return torch.tensor(0.0)

class Network(nn.Module):
    # ... (The Network class definition remains exactly the same) ...
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

    path_to_frames = r"C:\Users\Deborshi Chakrabarti\Desktop\try" 
    model_path = r'C:\Users\Deborshi Chakrabarti\Desktop\lava-dl-main\tutorials\lava\lib\dl\slayer\pilotnet\Trained\network.pt'
    results_path = r'C:\Users\Deborshi Chakrabarti\Desktop\New folder\pilotnet_sdnn\results.txt' # Path to your ground truth file

    # Load Image Frames
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

    # Load Ground Truth Data
    ground_truth_angles = []
    try:
        with open(results_path, 'r') as f:
            next(f) # Skip header line
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    ground_truth_angles.append(float(parts[1]))
        print(f"Loaded {len(ground_truth_angles)} ground truth angles.")
    except FileNotFoundError:
        print(f"Error: Ground truth file '{results_path}' not found.")
        exit()

    #  Load Model and Perform PGD 
    if sequence_tensor is not None:
        # Remember this part (taken from StackExchange)       
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
        
        epsilon = 0.03
        num_steps = 10
        alpha = epsilon / num_steps
        target_label = torch.tensor([[5.0]])
        
        original_input = sequence_tensor.unsqueeze(0)
        adversarial_input = original_input.clone().detach()

        for i in range(num_steps):
            adversarial_input.requires_grad = True
            output, _, _ = model(adversarial_input)
            criterion = nn.MSELoss()
            loss = criterion(output, target_label)
            model.zero_grad()
            loss.backward()
            grad = adversarial_input.grad.data
            adversarial_input = adversarial_input.detach() + alpha * grad.sign()
            eta = torch.clamp(adversarial_input - original_input, -epsilon, epsilon)
            adversarial_input = original_input + eta
            adversarial_input = torch.clamp(adversarial_input, -1, 1)

        print("Adversarial input generated via PGD.")

        # Get Final Predictions 
        with torch.no_grad():
            original_pred_tensor, _, _ = model(original_input)
            adversarial_pred_tensor, _, _ = model(adversarial_input)

        # Prepare data for plotting
        time_steps = np.arange(len(ground_truth_angles))
        original_pred_np = original_pred_tensor.squeeze().cpu().numpy()
        adversarial_pred_np = adversarial_pred_tensor.squeeze().cpu().numpy()

        # Ensure all arrays have the same length for plotting
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
        
        plot_filename = 'attack_comparison.png'
        plt.savefig(plot_filename)
        print(f"\nComparison plot saved as '{plot_filename}'")
        # To display the plot directly, uncomment the next line
        plt.show()