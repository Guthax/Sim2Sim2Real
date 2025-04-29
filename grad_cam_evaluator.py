from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cv2
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
from torch import cosine_similarity
from torch.backends.cudnn import deterministic

from util.grad_cam import grad_cam, feature_map, final_representation, compare_histograms, compare_image_histograms
from utils import lr_schedule
from skimage.metrics import structural_similarity as ssim, mean_squared_error

import torch.nn.functional as F

class ModelComparator:
    def __init__(self, model_1=None, model_2=None, video_path=None):
        #self.model_1 = PPO.load(model_1_path) if model_1_path else PPO("CnnPolicy", make_vec_env("CartPole-v1"),
        #                                                               verbose=0)
        #self.model_2 = PPO.load(model_2_path) if model_2_path else PPO("CnnPolicy", make_vec_env("CartPole-v1"),
        #                                                               verbose=0)

        self.video_path = video_path
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1.policy.eval()
        self.model_2.policy.eval()

    def run(self):
        if not self.video_path:
            raise ValueError("Video path not provided.")

        cap = cv2.VideoCapture(self.video_path)

        frame_count = 0
        total_ssim = 0
        total_am_sim = 0
        total_grad_hist_compare = 0
        total_kl = 0
        total_fr_similarity = 0
        total_action_diff = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            frame_processed = np.expand_dims(np.transpose(frame, (2,0,1)) / 255.0, axis=0)
            obs = {
                "camera_rgb": deepcopy(frame_processed),
                "vehicle_dynamics": [[0]],
            }

            fr= final_representation(self.model_1, obs, key="camera_rgb")

            grad_cam_model_1 = grad_cam(self.model_1, obs, key="camera_rgb", action=torch.tensor([0]))
            grad_cam_model_2 = grad_cam(self.model_2, obs, key="camera_rgb", action=torch.tensor([0]))
            cv2.imshow("grad1", grad_cam_model_1)
            cv2.imshow("grad2", grad_cam_model_2)
            cv2.waitKey(1)

            fm1 = feature_map(self.model_1, obs, key="camera_rgb").squeeze(0)
            fm2 = feature_map(self.model_2, obs, key="camera_rgb").squeeze(0)

            fr1 = final_representation(self.model_1, obs, key="camera_rgb")
            fr2 = final_representation(self.model_2, obs, key="camera_rgb")

            similarity = F.cosine_similarity(fr1, fr2, dim=1)
            total_fr_similarity += similarity
            assert fm1.shape == fm2.shape

            total_attention_map_sim_for_frame = 0
            for attention_map_index in range(fm1.shape[0]):
                am_1 = fm1[attention_map_index].detach().cpu().numpy()
                am_2 = fm2[attention_map_index].detach().cpu().numpy()
                similarity = ssim(am_1, am_2)
                total_attention_map_sim_for_frame += similarity
            avg_attention_map_for_frame = total_attention_map_sim_for_frame / fm1.shape[0]
            total_am_sim += avg_attention_map_for_frame

            img1_gray = cv2.cvtColor(grad_cam_model_1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(grad_cam_model_2, cv2.COLOR_RGB2GRAY)

            a1 = self.model_1.predict(obs, deterministic=True)
            a2 = self.model_2.predict(obs, deterministic=True)
            total_action_diff += abs(a2[0] - a1[0])

            tensor = obs_as_tensor(obs, device='cuda' if torch.cuda.is_available() else 'cpu')
            tensor = preprocess_obs(tensor, self.model_1.observation_space)

            dist1 = self.model_1.policy.get_distribution(tensor)
            dist2 = self.model_2.policy.get_distribution(tensor)

            total_kl += torch.distributions.kl_divergence(dist1.distribution, dist2.distribution).mean()
            # Compute SSIM
            score, _ = ssim(img1_gray, img2_gray, full=True)
            total_ssim += score

            overal, b, g, r = compare_image_histograms(grad_cam_model_1, grad_cam_model_2)
            total_grad_hist_compare += overal
            frame_count+=1
        print(f"AVG AM SIM: {total_am_sim / frame_count}")
        print(f"AVG GCAM SSIM: {total_ssim / frame_count}")
        print(f"AVG HIST COMPARE: {total_grad_hist_compare / frame_count}")
        print(f"AVG FINAL SIM : {total_fr_similarity / frame_count}")
        print(f"AVG ACTION DIFF : {total_action_diff / frame_count}")
        print(f"AVG KL DIVERGENCE : {total_kl / frame_count}")



model_1 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/duckie_rgb_baseline_long_model_trained_3200000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
model_2 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/random", device='cuda' if torch.cuda.is_available() else 'cpu')

params_1 = model_1.policy.parameters()
params_2 = model_2.policy.parameters()

# Convert parameters to arrays
params_1_flat = np.concatenate([p.flatten().detach().cpu().numpy() for p in params_1])
params_2_flat = np.concatenate([p.flatten().detach().cpu().numpy() for p in params_2])

# Compute cosine similarity
similarity = cosine_similarity(torch.Tensor([params_1_flat]), torch.Tensor([params_2_flat]))
print(f"General weight similarity: {similarity}")

ge = ModelComparator(video_path='/home/jurriaan/workplace/programming/Sim2Sim2Real/test/videos/duckiebot_real.mp4',
                      model_1=model_1,
                      model_2=model_2)
ge.run()