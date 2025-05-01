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

from util.grad_cam import grad_cam
from util.gradcam_targets import ContinuousActionTarget


class Evaluator:
    evaluation_environment: gym.Env

    def __init__(self, eval_env: gym.Env, algorithm: BaseAlgorithm, apply_grad_cam=False):
        self.evaluation_environment = eval_env
        self.algorithm = algorithm

        self.evaluation_timesteps = 100000
        self.episode_length = 100000

        self.apply_grad_cam = apply_grad_cam

    def evaluate(self, key=None):
        state =  self.evaluation_environment.reset()
        done = False
        timesteps = 0
        current_episode_length = 0
        total_reward = 0

        if self.apply_grad_cam:
            cv2.startWindowThread()
            cv2.namedWindow("gradients")

        while timesteps < self.evaluation_timesteps or self.evaluation_timesteps == -1:
            while not done and current_episode_length < self.episode_length:
                action, _states = self.algorithm.predict(state, deterministic=True)
                state, reward, done, info = self.evaluation_environment.step(action)
                #print(reward)

                if self.apply_grad_cam:
                    #self.grad_cam(state)
                    hm = grad_cam(self.algorithm, state, key="camera_rgb")
                    cv2.imshow("gradients", hm)
                    cv2.waitKey(1)
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