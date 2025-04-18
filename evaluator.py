import cv2
import gymnasium as gym
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor

from util.gradcam_targets import ContinuousActionTarget


class Evaluator:
    evaluation_environment: gym.Env

    def __init__(self, eval_env: gym.Env, algorithm: BaseAlgorithm, apply_grad_cam=False):
        self.evaluation_environment = eval_env
        self.algorithm = algorithm

        self.evaluation_timesteps = 100000
        self.episode_length = 100000

        self.apply_grad_cam = apply_grad_cam

    def evaluate(self, ):
        state =  self.evaluation_environment.reset()
        done = False
        timesteps = 0
        current_episode_length = 0
        total_reward = 0

        cv2.startWindowThread()
        cv2.namedWindow("gradients")

        while timesteps < self.evaluation_timesteps or self.evaluation_timesteps == -1:
            while not done and current_episode_length < self.episode_length:
                action, _states = self.algorithm.predict(state, deterministic=True)
                #print(f"Action: {action}")
                state, reward, done, info = self.evaluation_environment.step(action)
                #print(reward)
                if self.apply_grad_cam:
                    #self.grad_cam(state)
                    self.grad_cam(state, key="camera_rgb")
                    #self.grad_cam(state, key="camera_seg")
                total_reward += reward
                timesteps += 1
                current_episode_length += 1
            done = False
            state = self.evaluation_environment.reset()
            current_episode_length = 0
            avg_reward = total_reward / timesteps
            print(f"AVERAGE REWARD: {avg_reward}")

        avg_reward = total_reward / timesteps
        print(f"AVERAGE REWARD: {avg_reward}")

    def grad_cam(self, obs, key=None):
        #print(obs.shape)
        policy_net = self.algorithm.policy

        if key:
            last_cnn_layer= policy_net.features_extractor.extractors[key].cnn[0]
        else:
            last_cnn_layer = policy_net.features_extractor.cnn[1]

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
        tensor = preprocess_obs(tensor, self.algorithm.observation_space)
        #tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1).unsqueeze(0)

        with torch.set_grad_enabled(True):
            gaussian = self.algorithm.policy.get_distribution(tensor)
            action_distribution = gaussian.distribution
            action_sample = action_distribution.sample()  # Sample an action
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
        obs = np.transpose(obs, (2,1,0))
        heatmap = cv2.resize(gradcam, (obs.shape[0], obs.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        #overlay = cv2.addWeighted(obs, 0.5, heatmap, 0.5, 0)
        #overlay = cv2.resize(overlay, dsize=(480, 680))
        # Show the result
        #plt.figure(figsize=(10, 5))
        #plt.subplot(1, 2, 1)
        #plt.title("Original Observation")
        #plt.imshow(obs)

        #plt.subplot(1, 2, 2)
        #plt.title("Grad-CAM Visualization")
        if key:
            cv2.imshow(f"gradients - {key}", heatmap)
        else:
            cv2.imshow(f"gradients", heatmap)
        cv2.waitKey(1)
        # Remove hooks after use
        forward_handle.remove()
        backward_handle.remove()

        activations = {}
        gradients = {}


    def grad_cam2(self, obs):
        policy_net = self.algorithm.policy
        last_cnn_layer = policy_net.features_extractor.cnn[0]
        target_layers = [last_cnn_layer]

        tensor = obs_as_tensor(obs, device=torch.device("cpu"))
        #tensor = preprocess_obs(tensor, self.algorithm.observation_space)
        #tensor["camera_rgb"] = tensor["camera_rgb"].permute(2,0,1).unsqueeze(0)
        #input_tensor = tensor
        # Note: input_tensor can be a batch tensor with several images!

        # We have to specify the target we want to generate the CAM for.
        targets = [ContinuousActionTarget()]

        # Construct the CAM object once, and then re-use it on many images.
        with GradCAM(model=policy_net, target_layers=target_layers) as cam:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(obs, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            #model_outputs = cam.outputs
            cv2.imshow("Grad-CAM Visualization", visualization)

            cv2.waitKey(1)