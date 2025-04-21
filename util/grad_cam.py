import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
import cv2

def grad_cam(algorithm, obs, action=None, key=None):
    # print(obs.shape)
    policy_net = algorithm.policy

    if key:
        last_cnn_layer = policy_net.features_extractor.extractors[key].cnn[0]
    else:
        last_cnn_layer = policy_net.features_extractor.cnn[1]
    last_cnn_layer.eval()
    activations = {}
    gradients = {}

    # Hook for forarla_rgb_dynamics_model_trained_800000_stepsward pass (store activations)
    def forward_hook(module, input, output):
        activations["features"] = output

    # Hook for backward pass (store gradients)
    def backward_hook(module, grad_input, grad_output):
        gradients["features"] = grad_output[0]

    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)
    backward_handle = last_cnn_layer.register_backward_hook(backward_hook)

    tensor = obs_as_tensor(obs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    tensor = preprocess_obs(tensor, algorithm.observation_space)
    # tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1).unsqueeze(0)

    with torch.set_grad_enabled(True):
        gaussian = algorithm.policy.get_distribution(tensor)
        action_distribution = gaussian.distribution
        action_sample = action if action is not None else action_distribution.sample()  # Sample an action


        log_prob = action_distribution.log_prob(action_sample)  # Compute log probability

        # Compute gradients for Grad-CAM
        log_prob.sum().backward()

    activations = activations["features"].detach()
    gradients = gradients["features"].detach()

    # Compute Grad-CAM heatmap
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling
    gradcam = torch.relu(torch.sum(weights * activations, dim=1)).cpu().squeeze().numpy()  # Weighted sum
    # Normalize heatmap
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    # Resize heatmap to match original image
    if key:
        obs = obs[key]
    obs = obs.squeeze(0)
    obs = np.transpose(obs, (2, 1, 0))
    heatmap = cv2.resize(gradcam, (obs.shape[0], obs.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    # overlay = cv2.addWeighted(obs, 0.5, heatmap, 0.5, 0)
    # overlay = cv2.resize(overlay, dsize=(480, 680))
    # Show the result
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Observation")
    # plt.imshow(obs)

    # plt.subplot(1, 2, 2)
    # plt.title("Grad-CAM Visualization")
    #if key:
    #    cv2.imshow(f"gradients - {key}", heatmap)
    #else:
    #    cv2.imshow(f"gradients", heatmap)
    # Remove hooks after use

    forward_handle.remove()
    backward_handle.remove()

    activations = {}
    gradients = {}
    return heatmap


def feature_map(algorithm, obs, key=None):
    policy_net = algorithm.policy

    if key:
        last_cnn_layer = policy_net.features_extractor.extractors[key].cnn[0]
    else:
        last_cnn_layer = policy_net.features_extractor.cnn[1]
    last_cnn_layer.eval()
    activations = {}

    def forward_hook(module, input, output):
        activations["features"] = output

    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)

    tensor = obs_as_tensor(obs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    tensor = preprocess_obs(tensor, algorithm.observation_space)

    policy_net(tensor)

    feature_maps = activations["features"]


    return feature_maps
