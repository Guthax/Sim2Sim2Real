import math

import gymnasium as gym
import carla
import numpy as np
import random
import time
import cv2
from carla import LaneMarking, CityObjectLabel, Waypoint
from gymnasium import spaces
from jedi.inference.arguments import repack_with_argument_clinic

from simulators.carla.misc import get_pos, get_closest_waypoint, get_next_waypoint, compute_angle
from simulators.carla.route_planner import RoutePlanner


class SelfCarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, render=False):
        super(SelfCarlaEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town02_opt")

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.tick()
        self.client.reload_world(False)  # reload map keeping the world settings
        self.world.tick()
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        self.world.unload_map_layer(carla.MapLayer.Walls)


        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]
        self.actor_list = []
        self.render_mode = render
        self.collision_occurred = False
        self.offroad_occurred = False
        self.lane_invasion_occured = False
        self.route_planner = None

        self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1))
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 400, 3),
                                            dtype=np.uint8)  # Example observation (image input)

        self._setup_vehicle()

        self.count_until_randomization = 0
        self.randomize_every_steps = 20000

        self.world.tick()


    def _setup_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        valid_spawn_point_indexes = [4,10, 15,17, 28, 35,36, 41, 43, 45, 89, 95, 97 ]
        for _ in range(10):  # Try up to 10 times to find a valid spawn point
            spawn_point_index = random.choice(valid_spawn_point_indexes)
            spawn_point = spawn_points[spawn_point_index]
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")
        self.actor_list.append(self.vehicle)
        self.world.tick()
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_invasion_sensor()

    def _setup_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '400')
        camera_bp.set_attribute('image_size_y', '200')
        camera_bp.set_attribute('fov', '90')
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.image = None
        self.camera.listen(lambda image: self._process_image(image))

    def _setup_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(collision_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _setup_lane_invasion_sensor(self):
        invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        spawn_point = carla.Transform()
        self.invasion_sensor = self.world.spawn_actor(invasion_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.invasion_sensor)
        self.invasion_sensor.listen(self._on_lane_invasion)

    def _randomize_weather(self):
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 100),
            sun_altitude_angle=random.uniform(-90, 90),
            fog_density=random.uniform(0, 100)
        )
        self.world.set_weather(weather)

    def _randomize_time_of_day(self):
        weather = self.world.get_weather()
        weather.sun_altitude_angle = random.uniform(-90, 90)
        self.world.set_weather(weather)

    def _on_collision(self, event):
        #print("Collision detected!")
        self.collision_occurred = True

    def _on_lane_invasion(self, invasion_info):
        #penalized_lane_markings = [LaneMarking.Curb, LaneMarking.Grass, LaneMarking]
        types_crossed = [str(lane.type) for lane in invasion_info.crossed_lane_markings]
        #colors_crossed = [str(lane.color) for lane in invasion_info.crossed_lane_markings]
        #permissions = [str(lane.lane_change) for lane in invasion_info.crossed_lane_markings]
        #widths = [str(lane.width) for lane in invasion_info.crossed_lane_markings]
        #print(f"Lane invasion: {types_crossed},    {colors_crossed},    {permissions},   {widths}")
        #print(types_crossed)
        self.lane_invasion_occured = True

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.image = array

    def reset(self, *, seed=None, options=None):
        for actor in self.actor_list:
            actor.destroy()

        #if self.count_until_randomization >= self.randomize_every_steps:
        #    self._randomize_time_of_day()
        #    self._randomize_weather()
        #    self.count_until_randomization = 0
        self.image = np.zeros((200, 400, 3), dtype=np.uint8)
        self.actor_list = []

        self.collision_occurred = False
        self.offroad_occurred = False
        self.lane_invasion_occured = False
        self.previous_steer = None

        self._setup_vehicle()
        self.world.tick()
        self.route_planner = RoutePlanner(self.vehicle, 12)
        self.waypoints = self.route_planner.run_step()

        self.steps_alive = 0
        #if self.render_mode:
        #    self._draw_points()

        start_time = time.time()
        while self.image is None:
            if time.time() - start_time > 2.0:  # Timeout after 2 seconds
                print("Warning: No image received from camera sensor.")
                break
        self.world.tick()
        obs = self.image if self.image is not None else np.zeros((200, 400, 3), dtype=np.uint8)
        if self.render_mode:
            self.render()

        return obs, {}

    def _draw_points(self):
        life_time = 30
        for i in range(0, len(self.waypoints)-1):
            w0 = self.waypoints[i]
            w1 = self.waypoints[i+1]

            w0 = carla.Location(w0.location.x, w0.location.y, 0.25)
            w1 = carla.Location(w1.location.x, w1.location.y, 0.25)
            self.world.debug.draw_line(
                w0,
                w1,
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0, 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        #self.world.debug.draw_point(
        #    self.waypoints[-1][0], 0.1,
        #    carla.Color(0, 0, 255),
        #    life_time, False)

    def step(self, action):
        steer = action

        #target_speed = 4.77778  # 20 m/s, you can change this to any desired speed
        #transform = self.vehicle.get_transform()
        #forward_vector = transform.get_forward_vector()
        #target_velocity = forward_vector * target_speed

        # Set the target velocity (in the vehicle's local frame)
        #self.vehicle.set_target_velocity(target_velocity)

        target_speed = 5.5
        velocity = self.vehicle.get_velocity()
        current_speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5  # Convert to m/s

        # Simple proportional controller
        if current_speed < target_speed:
            throttle = min(0.5 + (target_speed - current_speed) * 0.1, 1.0)  # Increase throttle
        else:
            throttle = 0.0  # Cut throttle when speed is reached

        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)


        #self.vehicle.apply_control(carla.VehicleControl(throttle=float(0.5), steer=float(steer)))  # Fixed speed of 30 kph
        self.world.tick()

        observation = self.image if self.image is not None else np.zeros((84, 84, 3), dtype=np.uint8)
        self.waypoints = self.route_planner.run_step()

        # Calculate reward
        reward, done = self._get_reward_new()
        info = {}

        if self.render_mode:
            self.render()

        self.count_until_randomization += 1

        return observation, reward, done, False, info
    """
    def _get_reward(self):
        # Get the lateral distance from the center of the lane
        ego_loc = self.vehicle.get_transform().location
        wpt = get_closest_waypoint(self.waypoints, ego_loc.x, ego_loc.y)
        lane_distance, w = get_lane_dis(self.waypoints, ego_loc.x, ego_loc.y)  # Distance from lane center
        lane_distance = abs(lane_distance)
        #print(lane_distance)

        if lane_distance > 3:
            return -10.0, True  # Large negative reward and terminate episode

        # Reward for staying close to the lane center
        lane_penalty = max(2.5 - lane_distance, 0)

        # Get the steering action applied
        steer_value = self.vehicle.get_control().steer


        steer_change_penalty = -abs(steer_value - self.previous_steer) * 1.0 if self.previous_steer else 0
        self.previous_steer = steer_value  # Update previous steering value

        angle = (self.vehicle.get_transform().rotation.yaw - wpt[2]) % 360

        print("HEADING ALIGNMENT: ", angle)
        # Additional penalties for lane invasion
        invasion_penalty = -3.0  if self.lane_invasion_occured else 0

        # Collision is heavily penalized
        if self.collision_occurred:
            return -20.0, True  # Large negative reward and terminate episode

        # Reward is a combination of staying in lane, smooth steering, and avoiding sudden changes
        reward = 1 + lane_penalty + (0.1 * steer_change_penalty) + invasion_penalty
        #print(
        #    f"Lane penalty: {lane_penalty}, Steer change: {steer_change_penalty}, invasion_penalty: {invasion_penalty}, total: {reward}")

        done = False  # The episode continues unless a collision occurs

        return reward, done

    """
    def _get_reward_new(self):
        # Get the lateral distance from the center of the lane
        ego_loc = self.vehicle.get_transform().location
        waypt =  get_next_waypoint(self.waypoints, ego_loc.x, ego_loc.y, ego_loc.z)
        if waypt is None:
            waypt = get_closest_waypoint(self.waypoints, ego_loc.x, ego_loc.y, ego_loc.z)
            #print("No wpt")

        #self.world.debug.draw_point(
        #    carla.Location(waypt.transform.location.x, waypt.transform.location.y, 0.25), 0.1,
        #    carla.Color(255, 0, 0),
        #    20, False)

        lane_distance = abs(ego_loc.y - waypt.transform.location.y)
        lane_penalty = max(2.5 - lane_distance, 0)

        angle, dot_dir = compute_angle(ego_loc, waypt.transform.location, self.vehicle.get_transform().rotation.yaw)

        # Get the steering action applied
        steer_value = self.vehicle.get_control().steer
        steer_change_penalty = -abs(steer_value - self.previous_steer) * 1.0 if self.previous_steer else 0
        self.previous_steer = steer_value  # Update previous steering value
        invasion_penalty = 0

        # Collision is heavily penalized
        if self.collision_occurred:
            return -20.0, True  # Large negative reward and terminate episode

        # Reward is a combination of staying in lane, smooth steering, and avoiding sudden changes
        reward = 1.0  + dot_dir - lane_distance + steer_change_penalty + invasion_penalty
        #print(f"dot dir: {dot_dir}, lane dist: {lane_distance}, invasion: {invasion_penalty}, total: {reward}")

        if lane_distance > 3:
            reward = reward - 10.0
            return reward, True

        #print(
        #    f"Lane penalty: {lane_penalty}, Steer change: {steer_change_penalty}, invasion_penalty: {invasion_penalty}, total: {reward}")


        return reward, False

    def _get_reward_live(self):

        if self.collision_occurred:
            done = True
            reward = -100
            return reward, done
        if self.lane_invasion_occured:
            done = True
            reward = -20
            return reward, done

        reward = 1
        if self.steps_alive > 100:
            reward = 2

        if self.steps_alive > 300:
            reward = 3

        self.steps_alive += 1

        return reward, False

    def render(self, mode='human'):
        if self.image is not None:
            cv2.imshow("CARLA Camera", self.image)
            cv2.waitKey(1)

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        cv2.destroyAllWindows()
