import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
import cv2

def grad_cam(algorithm, obs, action=None, key=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # print(obs.shape)
    policy_net = algorithm.policy.to(device)

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

    tensor = obs_as_tensor(obs, device=device)
    tensor = preprocess_obs(tensor, algorithm.observation_space)
    # tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1).unsqueeze(0)

    with torch.set_grad_enabled(True):
        gaussian = algorithm.policy.get_distribution(tensor)
        action_distribution = gaussian.distribution
        action_sample = action.to(device) if action is not None else action_distribution.sample().to(device)  # Sample an action


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


def feature_map(algorithm, obs, key=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    policy_net = algorithm.policy.to(device)

    if key:
        last_cnn_layer = policy_net.features_extractor.extractors[key].cnn[2]
    else:
        last_cnn_layer = policy_net.features_extractor.cnn[1]
    last_cnn_layer.eval()
    activations = {}

    def forward_hook(module, input, output):
        activations["features"] = output

    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)

    tensor = obs_as_tensor(obs, device=device)
    tensor = preprocess_obs(tensor, algorithm.observation_space)

    policy_net(tensor)

    feature_maps = activations["features"]

    forward_handle.remove()

    activations = {}

    return feature_maps




def final_representation(algorithm, obs, key=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    policy_net = algorithm.policy.to(device)

    if key:
        last_cnn_layer = policy_net.features_extractor.extractors[key].linear
    else:
        last_cnn_layer = policy_net.features_extractor.cnn[-1]
    last_cnn_layer.eval()
    activations = {}

    def forward_hook(module, input, output):
        activations["features"] = output

    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)

    tensor = obs_as_tensor(obs, device=device)
    tensor = preprocess_obs(tensor, algorithm.observation_space)

    policy_net(tensor)

    feature_maps = activations["features"]


    forward_handle.remove()

    activations = {}

    return feature_maps



def calculate_histogram(image):
    """Calculates the normalized histogram for each color channel."""
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    norm_hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b
    norm_hist_g = hist_g / hist_g.sum() if hist_g.sum() > 0 else hist_g
    norm_hist_r = hist_r / hist_r.sum() if hist_r.sum() > 0 else hist_r

    return norm_hist_b, norm_hist_g, norm_hist_r

def compare_histograms(hist1, hist2):
    """Calculates the correlation between two histograms."""
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation

def compare_image_histograms(img1, img2):
    """Loads two images and compares their color histograms using correlation."""
    try:
        if img1 is None or img2 is None:
            print("Error: Could not open or find the images.")
            return None

        hist1_b, hist1_g, hist1_r = calculate_histogram(img1)
        hist2_b, hist2_g, hist2_r = calculate_histogram(img2)

        corr_b = compare_histograms(hist1_b, hist2_b)
        corr_g = compare_histograms(hist1_g, hist2_g)
        corr_r = compare_histograms(hist1_r, hist2_r)

        # You can average the correlations of the three channels for an overall similarity score
        overall_correlation = (corr_b + corr_g + corr_r) / 3

        return overall_correlation, corr_b, corr_g, corr_r

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
