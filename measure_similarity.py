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

from util.grad_cam import grad_cam, feature_map, final_representation, compare_histograms, compare_image_histograms, \
    final_representation_actor
from util.similarity import cca, gram_linear, feature_space_linear_cka, cka
from utils import lr_schedule
from skimage.metrics import structural_similarity as ssim, mean_squared_error

import torch.nn.functional as F



class SimilarityMeasurer:
    def __init__(self, model_1=None, model_2=None, video_path=None):

        self.video_path = video_path
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1.policy.eval()
        self.model_2.policy.eval()


    def run_rgb(self):
        if not self.video_path:
            raise ValueError("Video path not provided.")

        cap = cv2.VideoCapture(self.video_path)

        samples_m1 = []
        samples_m2 = []

        samples_action_m1 = []
        samples_action_m2 = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            frame_processed = np.expand_dims(np.transpose(frame, (2,0,1)) / 255.0, axis=0)
            obs = {
                "camera_rgb": deepcopy(frame_processed),
                "vehicle_dynamics": [[0]],
            }
            fr1 = final_representation(self.model_1, obs, key="camera_rgb").detach().cpu().numpy()
            fr2 = final_representation(self.model_2, obs, key="camera_rgb").detach().cpu().numpy()
            samples_m1.append(fr1[0])
            samples_m2.append(fr2[0])

            fr_a1 = final_representation_actor(self.model_1, obs).detach().cpu().numpy()
            fr_a2 = final_representation_actor(self.model_2, obs).detach().cpu().numpy()
            samples_action_m1.append(fr_a1[0])
            samples_action_m2.append(fr_a2[0])
        samples_m1 = np.array(samples_m1)
        samples_m2 = np.array(samples_m2)


        samples_action_m1 = np.array(samples_action_m1)
        samples_action_m2 = np.array(samples_action_m2)

        cca_score = cca(samples_m1, samples_m2)
        cka_from_examples = cka(gram_linear(samples_m1), gram_linear(samples_m2))
        cka_from_features = feature_space_linear_cka(samples_m1, samples_m2)

        print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
        print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
        print(f'CCA: {cca_score}')


        cca_score = cca(samples_action_m1, samples_action_m2)
        cka_from_examples = cka(gram_linear(samples_action_m1), gram_linear(samples_action_m2))
        cka_from_features = feature_space_linear_cka(samples_action_m1, samples_action_m2)

        print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
        print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
        print(f'CCA: {cca_score}')

    def run_gray(self):
        if not self.video_path:
            raise ValueError("Video path not provided.")

        cap = cv2.VideoCapture(self.video_path)

        samples_m1 = []
        samples_m2 = []

        samples_action_m1 = []
        samples_action_m2 = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_processed = np.expand_dims(np.expand_dims(frame / 255.0, axis=0), axis=0)
            obs = {
                "camera_gray": deepcopy(frame_processed),
                "vehicle_dynamics": [[0]],
            }
            fr1 = final_representation(self.model_1, obs, key="camera_gray").detach().cpu().numpy()
            fr2 = final_representation(self.model_2, obs, key="camera_gray").detach().cpu().numpy()
            samples_m1.append(fr1[0])
            samples_m2.append(fr2[0])

            fr_a1 = final_representation_actor(self.model_1, obs).detach().cpu().numpy()
            fr_a2 = final_representation_actor(self.model_2, obs).detach().cpu().numpy()
            samples_action_m1.append(fr_a1[0])
            samples_action_m2.append(fr_a2[0])
        samples_m1 = np.array(samples_m1)
        samples_m2 = np.array(samples_m2)


        samples_action_m1 = np.array(samples_action_m1)
        samples_action_m2 = np.array(samples_action_m2)

        cca_score = cca(samples_m1, samples_m2)
        cka_from_examples = cka(gram_linear(samples_m1), gram_linear(samples_m2))
        cka_from_features = feature_space_linear_cka(samples_m1, samples_m2)

        print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
        print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
        print(f'CCA: {cca_score}')


        cca_score = cca(samples_action_m1, samples_action_m2)
        cka_from_examples = cka(gram_linear(samples_action_m1), gram_linear(samples_action_m2))
        cka_from_features = feature_space_linear_cka(samples_action_m1, samples_action_m2)

        print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
        print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
        print(f'CCA: {cca_score}')



model_1 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_gray_domain_rand", device='cuda' if torch.cuda.is_available() else 'cpu')
model_2 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_gray_no_domain_rand", device='cuda' if torch.cuda.is_available() else 'cpu')

ge = SimilarityMeasurer(video_path='/home/jurriaan/workplace/programming/Sim2Sim2Real/test/videos/duckie_video_right_lane.mp4',
                      model_1=model_1,
                      model_2=model_2)
ge.run_gray()
