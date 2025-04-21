"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import os


from PIL import Image
from platform import architecture

import numpy as np
import torch.cuda
import pyglet
import pyglet.window.key as key  # ✅ Corrected import
from contracts.library.extensions import kwarg
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC, TD3
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from evaluator import Evaluator
from envs.duckietown.duckietown import DuckietownBaseDynamics
import cv2

# Initialize the environment
env = DuckietownBaseDynamics()
env.reset()
env.render()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'mp4v' for .mp4 output
video_writer = cv2.VideoWriter("test/videos/duckie_video_left_lane.mp4", fourcc, 20, (160, 120))


# Register keyboard press handler
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        video_writer.release()

        sys.exit(0)
    # Optional: Screenshot (requires skimage)
    # elif symbol == key.RETURN:
    #     print('Saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register key state handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# Main control update loop
def update(dt):
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0])

    if key_handler[key.UP]:
        action = np.array([0.0])
    if key_handler[key.DOWN]:
        action = np.array([0.0])
    if key_handler[key.LEFT]:
        action = np.array([-0.5])
    if key_handler[key.RIGHT]:
        action = np.array([0.5])
    if key_handler[key.SPACE]:
        action = np.array([0.0])

    #v1 = action[0]
    #v2 = action[1]

    # Limit radius of curvature
    #if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
    #    delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
    #    v1 += delta_v
    #    v2 -= delta_v

    #action[0] = v1
    #action[1] = v2

    # Speed boost
    #if key_handler[key.LSHIFT]:
    #    action *= 1.5

    obs, reward, done, _, info = env.step(action)
    #print(f"step_count = {env.unwrapped.step_count}, reward = {reward:.3f}")

    video_writer.write(obs["camera_rgb"])  # Write the frame to video
    # Optional screenshot
    if key_handler[key.RETURN]:
        im = Image.fromarray(obs["camera_rgb"])
        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()

# Schedule the update function to be called at each frame
pyglet.clock.schedule_interval(update, 1.0 / 20)

# Enter the Pyglet event loop
pyglet.app.run()

# Cleanup
env.close()