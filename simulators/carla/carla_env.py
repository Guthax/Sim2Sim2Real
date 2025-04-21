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
from simulators.carla.route_planner import RoutePlanner

import matplotlib.pyplot as plt
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

def slight_variation(base, delta):
    return base + random.uniform(-delta, delta)
class SelfCarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, rgb_camera=True, seg_camera=False, render=False, domain_rand= True, layered_mapping=False, convert_segmentation=True):
        super(SelfCarlaEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town02_opt")

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


        self.layered_mapping = layered_mapping
        self.layered_mapping_counter = 200000
        self.current_mapping_counter = 0
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
        self.weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.CloudyNight,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.SoftRainSunset,
            carla.WeatherParameters.SoftRainNoon,
        ]
        self.distance_until_lap_complete = 5
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



    def _setup_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()

        #valid_spawn_point_indexes = [10, 15, 97, 95, 33, 41, 1, 86, 87, 89]
        valid_spawn_point_indexes = [6,11,15,95]
        #valid_spawn_point_indexes = [10,15, 28,89, 35,43,97, 20, 23, 95,33, 39]
        for _ in range(10):  # Try up to 10 times to find a valid spawn point
            spawn_point_index = random.choice(valid_spawn_point_indexes)
            spawn_point = spawn_points[spawn_point_index]
            print(f"Spawn point: {spawn_point_index}")
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
        spawn_point = carla.Transform(carla.Location(x=2.5, z=1.3))
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

    def _randomize_weather(self):
        self.world.set_weather(random.choice(self.weather_presets))

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
                cloudiness=random.uniform(0.0, 80.0),  # Avoid fully overcast
                precipitation=random.uniform(0.0, 30.0),  # Light rain only
                wind_intensity=random.uniform(0.0, 40.0),
                sun_azimuth_angle=random.uniform(0.0, 360.0),
                sun_altitude_angle=random.uniform(15.0, 75.0),  # Keep sun up
                fog_density=random.uniform(0.0, 20.0),  # Light fog
                fog_distance=random.uniform(50.0, 500.0),  # Allow visibility
                wetness=random.uniform(0.0, 50.0),  # Slightly wet roads
                fog_falloff=random.uniform(0.8, 5.0),
                scattered_light_intensity=random.uniform(10.0, 80.0)
            )
            self.world.set_weather(weather)
            self.count_until_randomization = 0

        if self.current_mapping_counter > self.layered_mapping_counter and len(self.map_layers) > 0 and self.layered_mapping:
            layer_to_load = self.map_layers.pop(random.randrange(len(self.map_layers)))
            self.world.load_map_layer(layer_to_load)
            print("Loading: ", layer_to_load)
            self.current_mapping_counter = 0
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
        #if self.turn_on_render:
        #     self._draw_points()

        start_time = time.time()



        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.image_rgb = self.rgb_queue.get()
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "camera_seg": self.image_seg,
                "vehicle_dynamics": 0.0,
            }
        elif self.camera_rgb_enabled:
            self.image_rgb = self.rgb_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "vehicle_dynamics": 0.0
            }

        elif self.camera_seg_enabled:
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_seg": self.image_seg,
                "vehicle_dynamics": 0.0
            }
        else:
            print("black")
            observation = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        self.current_steps = 0
        print(f"Completed laps: {self.laps_completed}, Laps done: {self.laps_done}")
        #self.laps_done += 1
        return observation, {}

    def _draw_points(self):
        life_time = 30
        for i in range(0, len(self.waypoints)-1):
            w0 = self.waypoints[i].transform
            w1 = self.waypoints[i+1].transform

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


        #self.vehicle.apply_control(carla.VehicleControl(throttle=float(0.5), steer=float(steer)))  # Fixed speed of 30 kph
        self.world.tick()

        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.image_rgb = self.rgb_queue.get()
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "camera_seg": self.image_seg,
                "vehicle_dynamics": steer,
            }
        elif self.camera_rgb_enabled:
            self.image_rgb = self.rgb_queue.get()
            observation = {
                "camera_rgb": self.image_rgb,
                "vehicle_dynamics": steer
            }

        elif self.camera_seg_enabled:
            self.image_seg = self.seg_queue.get()
            observation = {
                "camera_seg": self.image_seg,
                "vehicle_dynamics": steer
            }
        else:
            observation = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        self.waypoints = self.route_planner.run_step()

        # Calculate reward
        reward, done = self._get_reward_new()
        #print(reward)
        info = {}

        if self.turn_on_render:
            self.render()


        self.count_until_randomization += 1
        distance_to_spawn = self.vehicle.get_transform().location.distance(self.spawn_position.location)
        if distance_to_spawn < self.distance_until_lap_complete and self.current_steps >= self.min_steps_for_lap:
            done = True
            self.laps_completed += 1
            # Add a small delay for frame rate control
        if done:
            self.laps_done += 1
        self.current_steps += 1
        self.current_mapping_counter += 1
        self.total_amount_steps += 1

        return observation, reward, done, False, info

    def _get_reward_new(self):
        # Get the lateral distance from the center of the lane

        # Collision is heavily penalized
        if self.collision_occurred:
            return -20.0, True  # Large negative reward and terminate episode

        ego_loc = self.vehicle.get_transform().location

        waypt =  get_next_waypoint(self.waypoints, ego_loc.x, ego_loc.y, ego_loc.z)
        waypt = waypt if waypt else get_closest_waypoint(self.waypoints, ego_loc.x, ego_loc.y, ego_loc.z)
        #self.world.debug.draw_point(
        #    carla.Location(waypt.transform.location.x, waypt.transform.location.y, 0.25), 0.1,
        #    carla.Color(255, 0, 0),
         #   20, False)

        lane_distance = abs(ego_loc.y - waypt.transform.location.y)
        lane_penalty = max(2.5 - lane_distance, 0)

        angle, dot_dir = compute_angle(ego_loc, waypt.transform.location, self.vehicle.get_transform().rotation.yaw)

        # Get the steering action applied
        steer_value = self.vehicle.get_control().steer
        steer_change_penalty = -abs(steer_value - self.previous_steer) * 0.5 if self.previous_steer else 0
        self.previous_steer = steer_value  # Update previous steering value
        invasion_penalty = 0
        steer_change_penalty = 0


        # Reward is a combination of staying in lane, smooth steering, and avoiding sudden changes
        reward = 1.0  + dot_dir - lane_distance + steer_change_penalty + invasion_penalty

        project_camera = self.camera_rgb if self.camera_rgb_enabled else self.camera_seg
        is_off_road = self.world.get_map().get_waypoint(project_camera.get_transform().location, project_to_road=False) is None

        if is_off_road:
            reward = reward - 10.0
            return reward, True
        #if self.lane_invasion_occured:
        #    return reward - 5, Tru
        if lane_distance > 5:
            return reward, True

        #print(
        #    f"Lane penalty: {lane_distance}, Dot dir: {dot_dir}, Steer change: {steer_change_penalty}, invasion_penalty: {invasion_penalty}, total: {reward}")


        return reward, False

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
