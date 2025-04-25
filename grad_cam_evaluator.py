from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cv2
from torch.backends.cudnn import deterministic

from util.grad_cam import grad_cam, feature_map, final_representation
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
            cv2.waitKey(10)

            fm1 = feature_map(self.model_1, obs, key="camera_rgb").squeeze(0)
            fm2 = feature_map(self.model_2, obs, key="camera_rgb").squeeze(0)

            fr1 = final_representation(self.model_1, obs, key="camera_rgb")
            fr2 = final_representation(self.model_2, obs, key="camera_rgb")

            similarity = F.cosine_similarity(fr1, fr2, dim=1)
            print("Similarity: ", similarity)
            assert fm1.shape == fm2.shape

            total_attention_map_sim_for_frame = 0
            for attention_map_index in range(fm1.shape[0]):
                am_1 = fm1[attention_map_index].detach().cpu().numpy()
                am_2 = fm2[attention_map_index].detach().cpu().numpy()
                similarity = ssim(am_1, am_2)
                total_attention_map_sim_for_frame += similarity
            avg_attention_map_for_frame = total_attention_map_sim_for_frame / fm1.shape[0]
            total_am_sim += avg_attention_map_for_frame
            print(avg_attention_map_for_frame)
            print(f"{fm1.shape}, {fm2.shape}")
            total_ssim += ssim(grad_cam_model_1, grad_cam_model_2, channel_axis=-1)
            frame_count+=1
        print(f"AVG AM SIM: {total_am_sim / frame_count}")
        print(f"AVG GCAM SSIM: {total_ssim / frame_count}")
model_1 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_rgb_heavy_domain_rand_3_model_trained_2000000", device='cuda' if torch.cuda.is_available() else 'cpu')
model_2 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/duckie_rgb_model_trained_600000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
ge = ModelComparator(video_path='/home/jurriaan/workplace/programming/Sim2Sim2Real/test/videos/duckiebot_real.mp4',
                      model_1=model_1,
                      model_2=model_2)
ge.run()