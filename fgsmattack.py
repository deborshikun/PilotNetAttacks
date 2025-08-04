import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms

import lava.lib.dl.slayer as slayer


# A placeholder for the loss function needed by the model definition.
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
                # count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())
                count.append(torch.sum((x[..., 1:] > 0).to(x.dtype)).item())              #deborshi hack
        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)


if __name__ == "__main__":
    # Load Frames from Folder 
    path_to_frames = r"C:\Users\Deborshi Chakrabarti\Desktop\try" 
    model_path = r'C:\Users\Deborshi Chakrabarti\Desktop\lava-dl-main\tutorials\lava\lib\dl\slayer\pilotnet\Trained\network.pt'
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
        else:
            print("No frames found in the directory.")
    except FileNotFoundError:
        print(f"Error: The directory '{path_to_frames}' was not found.")


    if sequence_tensor is not None:
        # Load the Pre-trained Model        
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

        # Perform FGSM Attack
        epsilon = 0.02
        target_label = torch.tensor([[5.0]]) # Example true label
        
        # Prepare input: add batch dimension and enable gradient calculation
        original_input = sequence_tensor.unsqueeze(0)
        original_input.requires_grad = True

        output, _, _ = model(original_input)
        criterion = nn.MSELoss()
        loss = criterion(output, target_label) # Compares a 201-element tensor to a 1-element tensor
        
        model.zero_grad()
        loss.backward()
        
        gradient_sign = original_input.grad.data.sign()
        adversarial_input = original_input + epsilon * gradient_sign
        print(" Adversarial input generated.")

        output_folder = r'C:\Users\Deborshi Chakrabarti\Desktop\advattk\attack_results_fgsm'
        os.makedirs(output_folder, exist_ok=True)
        
        # Remove the batch dimension to iterate through frames
        adversarial_frames_tensor = adversarial_input.squeeze(0)
        to_pil_image = transforms.ToPILImage()
        
        num_frames = adversarial_frames_tensor.shape[3] # Get number of time steps
        for i in range(num_frames):
            # Get the tensor for the i-th frame
            frame_as_tensor = adversarial_frames_tensor[:, :, :, i]
            
            # Convert tensor to a PIL Image
            image = to_pil_image(frame_as_tensor)
            
            # Save the image
            image_filename = os.path.join(output_folder, f'adversarial_frame_{i:04d}.png')
            image.save(image_filename)
            
        print(f"Adversarial frames saved to '{output_folder}' directory.")
        
        #  Compare Predictions 
        print("\nRunning inference...")
        with torch.no_grad():
            original_pred, _, _ = model(original_input)
            adversarial_pred, _, _ = model(adversarial_input)

        # print(f"\nOriginal Prediction: {original_pred.item():.4f}")
        # print(f"Adversarial Prediction: {adversarial_pred.item():.4f}")
        print(f"\nOriginal Prediction: {original_pred.mean().item():.4f}")
        print(f"Adversarial Prediction: {adversarial_pred.mean().item():.4f}")

        if abs(original_pred.mean().item() - adversarial_pred.mean().item()) > epsilon * 2:
            print("\n Attack successful! The prediction changed significantly.")
        else:
            print("\n Attack may not have been strong enough to change the prediction.")