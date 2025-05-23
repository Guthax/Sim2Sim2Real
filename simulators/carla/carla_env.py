import math
import queue

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
from simulators.carla.route_planner import RoutePlanner, is_waypoint_behind_vehicle

import matplotlib.pyplot as plt


CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

def slight_variation(base, delta):
    return base + random.uniform(-delta, delta)
class SelfCarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, rgb_camera=True, seg_camera=False, render=False, domain_rand= True, map_change=False, convert_segmentation=True):
        super(SelfCarlaEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town02_opt")
        self.current_map_name = "Town02_opt"

        fps = 20
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1 / fps
        settings.synchronous_mode = True

        settings.substepping = True
        settings.max_substep_delta_time = 0.0166666666666667
        settings.max_substeps = 3
        self.world.apply_settings(settings)
        self.world.tick()
        self.client.reload_world(False)  # reload map keeping the world settings
        self.world.tick()


        self.map_change = map_change
        self.map_change_counter = 1000
        self.current_map_change_counter = 0
        self.map_layers = [carla.MapLayer.Buildings, carla.MapLayer.Decals,
                  carla.MapLayer.Foliage, carla.MapLayer.ParkedVehicles,
                  carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights,
                  carla.MapLayer.Walls]
        [self.world.unload_map_layer(layer) for layer in self.map_layers]


        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]
        self.actor_list = []
        self.turn_on_render = render
        self.collision_occurred = False
        self.offroad_occurred = False
        self.lane_invasion_occured = False
        self.route_planner = None

        self.rgb_queue = queue.Queue()
        self.seg_queue = queue.Queue()

        self.image_rgb = None
        self.image_seg = None

        self.spawn_position = None
        self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1))

        self.camera_rgb_enabled = rgb_camera
        self.camera_seg_enabled = seg_camera

        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.observation_space = spaces.Dict({
                "camera_rgb": spaces.Box(low=0, high=255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8),
                "camera_seg": spaces.Box(low=0, high=1, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8),
                "vehicle_dynamics": spaces.Box(low=np.float32(-1), high=np.float32(1))
            })
        else:
            camera_space_key = "camera_rgb" if self.camera_rgb_enabled else "camera_seg"
            low = 0
            high = 255 if self.camera_rgb_enabled else 1
            self.observation_space = spaces.Dict({
                camera_space_key: spaces.Box(low=low, high=high, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8),
                "vehicle_dynamics": spaces.Box(low=np.float32(-1), high=np.float32(1))
            })

        self._setup_vehicle()

        self.domain_rand = domain_rand


        self.count_until_randomization = 0
        self.randomize_every_steps = 1

        self.distance_until_lap_complete = 20
        self.min_steps_for_lap = 600
        self.current_steps = 0


        self.laps_completed = 0
        self.laps_done = 0

        self.total_amount_steps = 0

        #weather = carla.WeatherParameters(
        #    wetness=50,
        #)
        #self.world.set_weather(weather)
        

        #self.world.tick()

        self.convert_segmentation = convert_segmentation

        if not self.domain_rand:
            self.world.set_weather(carla.WeatherParameters.CloudySunset)



    def _setup_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()

        #valid_spawn_point_indexes = [10, 15, 97, 95, 33, 41, 1, 86, 87, 89]

        #valid_spawn_point_indexes = [15,95]
        valid_spawn_point_indexes = {
            "Town01_opt": [207, 202,197, 184, 118,174,47, 54, 108],
            #"Town02_opt": [28,89,4,45,78,41,43,35,1,33,95,36,9,10,14,15,97],
            "Town02_opt": [89,4,87,28, 97, 22, 15, 17, 14, 10,36,95,9,39,41,43,1, 78]
            #"Town02_opt": [4,22, 9,41, 1, 15],
        }

        for _ in range(10):  # Try up to 10 times to find a valid spawn point
            spawn_point_index = random.choice(valid_spawn_point_indexes[self.current_map_name])
            spawn_point = spawn_points[spawn_point_index]
            print(f"Spawn point: {spawn_point_index}")
            angle_deviance_radians = random.uniform(-math.pi /12, math.pi / 12)
            angle_deviance_degrees = math.degrees(angle_deviance_radians)
            print("Deviance: ", angle_deviance_degrees)
            #spawn_point.rotation.yaw += angle_deviance_degrees

            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)

            self.spawn_position = spawn_point
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")

        self.actor_list.append(self.vehicle)
        self.world.tick()

        self._setup_cameras()

        self._setup_collision_sensor()
        self._setup_lane_invasion_sensor()


    def _setup_cameras(self):

        if self.camera_seg_enabled:
            self._setup_camera_seg()
        if self.camera_rgb_enabled:
            self._setup_camera_rgb()

    def _setup_camera_rgb(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        #camera_bp.set_attribute('enable_postprocess_effects', 'True')
        #camera_bp.set_attribute("sensor_tick", "0.05")  # Match world tick
        spawn_point = carla.Transform(carla.Location(x=2.4, z=1.25), carla.Rotation(pitch=-20))
        self.camera_rgb = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)
        self.camera_rgb.listen(lambda image: self._process_image_rgb(image))


    def _setup_camera_seg(self):
        camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        #camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        #camera_bp.set_attribute("sensor_tick", "0.05")  # Match world tick
        spawn_point = carla.Transform(carla.Location(x=2.4, z=1.25), carla.Rotation(pitch=-20))
        self.camera_seg = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_seg)
        self.camera_seg.listen(lambda image: self._process_image_seg(image))

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

    def _process_image_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3] #BGR
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB) #RGB
        self.rgb_queue.put(array)

    def _process_image_seg(self, image):
        #print("Segmentation image received")
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3] #BGR
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB) #RGB
        self.seg_queue.put(array)

    def reset(self, *, seed=None, options=None):
        for actor in self.actor_list:
            actor.destroy()

        if self.count_until_randomization >= self.randomize_every_steps and self.domain_rand:
            weather = carla.WeatherParameters(
                cloudiness=random.uniform(0.0, 80.0),
                wind_intensity=random.uniform(0.0, 40.0),
                sun_altitude_angle=random.uniform(5.0, 30.0),  # Keep sun up
                fog_density=random.uniform(0.0, 60.0),  # Light fog
                fog_distance=random.uniform(20.0, 100.0),  # Allow visibility
                wetness=random.uniform(0.0, 90.0)
            )
            self.world.set_weather(weather)
            self.count_until_randomization = 0

        if self.current_map_change_counter > self.map_change_counter and self.map_change:
            self.current_map_name = "Town02_opt" if self.current_map_name == "Town01_opt" else "Town01_opt"
            self.world = self.client.load_world(self.current_map_name)

            self.map_layers = [carla.MapLayer.Buildings, carla.MapLayer.Decals,
                               carla.MapLayer.Foliage, carla.MapLayer.ParkedVehicles,
                               carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights,
                               carla.MapLayer.Walls]
            [self.world.unload_map_layer(layer) for layer in self.map_layers]
            self.world.tick()
            self.current_map_change_counter = 0

        self.actor_list = []

        self.collision_occurred = False
        self.offroad_occurred = False
        self.lane_invasion_occured = False
        self.previous_steer = None

        self._setup_vehicle()
        self.world.tick()


        self.steps_alive = 0
        #if self.turn_on_render:
        #     self._draw_points()




        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.image_rgb = self.rgb_queue.get()
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "camera_seg": self.image_seg,
                "vehicle_dynamics": [0.0],
            }
        elif self.camera_rgb_enabled:
            self.image_rgb = self.rgb_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "vehicle_dynamics": [0.0]
            }

        elif self.camera_seg_enabled:
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_seg": self.image_seg,
                "vehicle_dynamics": [0.0]
            }
        else:
            print("black")
            observation = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        self.current_steps = 0
        print(f"Completed laps: {self.laps_completed}, Laps done: {self.laps_done}")
        #self.laps_done += 1
        #print(f"Vehicle before calling routeplanner: {self.vehicle.get_transform()}")
        self.route_planner = RoutePlanner(self.vehicle)
        #print(f"Vehicle after calling routeplanner: {self.vehicle.get_transform()}")
        #print(f"First waypoint: {self.route_planner.waypoints_queue[0]}")
        distance = self.route_planner.locations[0].distance(self.vehicle.get_location())
        #for waypt in self.route_planner.waypoints:
        #    self.world.debug.draw_point(
        #    carla.Location(waypt.transform.location.x, waypt.transform.location.y, 0.25), 0.1,
        #        carla.Color(0, 255, 0),
        #        2, False)

        #distance = self.route_planner.waypoints_queue[0].location.distance(self.vehicle.get_transform().get_location())

        return observation, {}


    def step(self, action):
        steer = action

        target_speed = 7
        velocity = self.vehicle.get_velocity()
        current_speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5  # Convert to m/s

        # Simple proportional controller
        if current_speed < target_speed:
            throttle = min(0.5 + (target_speed - current_speed) * 0.1, 1.0)  # Increase throttle
        else:
            throttle = 0.0  # Cut throttle when speed is reached

        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)


        self.world.tick()

        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.image_rgb = self.rgb_queue.get()
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "camera_seg": self.image_seg,
                "vehicle_dynamics": [0.0],
            }
        elif self.camera_rgb_enabled:
            self.image_rgb = self.rgb_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "vehicle_dynamics": [0.0]
            }

        elif self.camera_seg_enabled:
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_seg": self.image_seg,
                "vehicle_dynamics": [0.0]
            }
        else:
            observation = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)


        # Calculate reward
        reward, done = self._get_reward_norm()
        #print(reward)
        info = {}

        if self.turn_on_render:
            self.render()


        self.count_until_randomization += 1
        distance_to_spawn = self.vehicle.get_transform().location.distance(self.spawn_position.location)
        #print(distance_to_spawn)
        if distance_to_spawn < self.distance_until_lap_complete and self.current_steps >= self.min_steps_for_lap:
            done = True
            self.laps_completed += 1
            # Add a small delay for frame rate control
        if done:
            self.laps_done += 1
        self.current_steps += 1
        self.current_map_change_counter += 1
        self.total_amount_steps += 1
        return observation, reward, done, False, info

    def _get_reward_norm(self):
        # Get the lateral distance from the center of the lane

        # Collision is heavily penalized
        if self.collision_occurred:
            return -1, True


        ego_loc = self.vehicle.get_transform().location
        waypt =  self.route_planner.waypoints[0]
        #print(f"First waypoint in step: {waypt}")
        if is_waypoint_behind_vehicle(self.vehicle.get_transform(), waypt.transform):
            self.route_planner.waypoints.pop(0)
            waypt = self.route_planner.waypoints[0]
        #waypt = waypt if waypt else get_closest_waypoint(self.waypoints, ego_loc.x, ego_loc.y, ego_loc.z)


        #self.world.debug.draw_point(
        #    carla.Location(waypt.transform.location.x, waypt.transform.location.y, 0.25), 0.1,
        #    carla.Color(255, 0, 0),
        #    20, False)
        lane_distance = abs(ego_loc.y - waypt.transform.location.y)
        #print(lane_distance)
        angle, dot_dir = compute_angle(ego_loc, waypt.transform.location, self.vehicle.get_transform().rotation.yaw)

        project_camera = self.camera_rgb if self.camera_rgb_enabled else self.camera_seg
        is_off_road = self.world.get_map().get_waypoint(project_camera.get_transform().location, project_to_road=False) is None

        if is_off_road or lane_distance > 5:
            reward = -1
            return reward, True

        lane_penalty = min(lane_distance / 5.0, 1.0)  # in [0, 1]
        dot_penalty = (1 - dot_dir) / 2.0
        reward = 1.0 - 0.8 * lane_penalty - 0.2 * dot_penalty  # max = 1.0, min ~ 0.0


        return reward, False
        #print(
        #    f"Lane penalty: {lane_distance}, Dot dir: {dot_dir}, Steer change: {steer_change_penalty}, invasion_penalty: {invasion_penalty}, total: {reward}")


    def render(self, mode='human'):
        if self.image_rgb is not None:

            cv2.imshow("CARLA Camera RGB", self.image_rgb)
            cv2.waitKey(1)


        if self.image_seg is not None:
            cv2.imshow("CARLA Camera SEG", self.image_seg)
            cv2.waitKey(1)

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        cv2.destroyAllWindows()
