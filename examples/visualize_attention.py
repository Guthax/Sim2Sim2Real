import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import obs_as_tensor
from torch.types import Device

from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from simulators.duckietown.wrappers import ResizeWrapper, CropWrapper, CannyWrapper

# Load environment
env = DuckietownBaseDynamics(render_img=True, randomize_maps_on_reset=True)
env = ResizeWrapper(env)
env = CropWrapper(env)

# Reset env and get observation
obs = env.reset()

# Load PPO model
model = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_only_rgb_domain_rand_model_trained_1000000_steps.zip')
policy_net = model.policy

# Get last CNN layer (modify if using a different architecture)
last_cnn_layer = policy_net.features_extractor.extractors.camera_rgb.cnn[-2]

# Store activations and gradients
cv2.startWindowThread()
cv2.namedWindow("gradients")

done = False
while True:
    activations = {}
    gradients = {}

    # Hook for forward pass (store activations)
    def forward_hook(module, input, output):
        activations["features"] = output

    # Hook for backward pass (store gradients)
    def backward_hook(module, grad_input, grad_output):
        gradients["features"] = grad_output[0]



    # Register hooks
    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)
    backward_handle = last_cnn_layer.register_backward_hook(backward_hook)

    tensor = obs_as_tensor(obs, device=torch.device("cpu"))
    tensor = preprocess_obs(tensor, model.observation_space)
    tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1).unsqueeze(0)
    #tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1)

    with torch.set_grad_enabled(True):
        gaussian = model.policy.get_distribution(tensor)
        action_distribution = gaussian.distribution
        action_sample = action_distribution.sample()  # Sample an action
        log_prob = action_distribution.log_prob(action_sample)  # Compute log probability

        # Compute gradients for Grad-CAM
        log_prob.sum().backward()  # Ensure backward() is called on a proper tensor

    # Get stored activations and gradients
    activations = activations["features"].detach()
    gradients = gradients["features"].detach()

    # Compute Grad-CAM heatmap
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling
    gradcam = torch.relu(torch.sum(weights * activations, dim=1)).squeeze().numpy()  # Weighted sum
    # Normalize heatmap
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    # Resize heatmap to match original image
    obs = obs["camera_rgb"]
    heatmap = cv2.resize(gradcam, (600,400))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(obs, 0.5, heatmap, 0.5, 0)

    # Show the result
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #plt.title("Original Observation")
    #plt.imshow(obs)

    #plt.subplot(1, 2, 2)
    #plt.title("Grad-CAM Visualization")
    cv2.imshow("gradients", overlay)

    cv2.waitKey(10)
    # Remove hooks after use
    forward_handle.remove()
    backward_handle.remove()

    obs, reward, done, info = env.step(action_sample)
