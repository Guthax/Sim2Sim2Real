import gymnasium as gym
import carla
import numpy as np
import random
import time
import cv2
from gymnasium import spaces


class SelfCarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, render=False):
        super(SelfCarlaEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town02_opt")

        self.client.reload_world(False)  # reload map keeping the world settings
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

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)  # Only Steering
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 400, 3),
                                            dtype=np.uint8)  # Example observation (image input)

        self._setup_vehicle()

    def _setup_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(10):  # Try up to 10 times to find a valid spawn point
            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")
        self.actor_list.append(self.vehicle)
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

    def _on_collision(self, event):
        print("Collision detected!")
        self.collision_occurred = True

    def _on_lane_invasion(self, invasion_info):
        print("Lane invasion detected!")
        self.lane_invasion_occured = True

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.image = array

    def reset(self, *, seed=None, options=None):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

        self.collision_occurred = False
        self.offroad_occurred = False
        self.lane_invasion_occured = False

        self._setup_vehicle()

        start_time = time.time()
        while self.image is None:
            if time.time() - start_time > 2.0:  # Timeout after 2 seconds
                print("Warning: No image received from camera sensor.")
                break

        obs = self.image if self.image is not None else np.zeros((84, 84, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):

        steer = action[0]
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=float(steer)))  # Fixed speed of 30 kph
        self.world.tick()
        observation = self.image if self.image is not None else np.zeros((84, 84, 3), dtype=np.uint8)

        # Calculate reward
        reward = self._calculate_reward()
        done = self.collision_occurred or self.offroad_occurred
        info = {}

        if self.collision_occurred or self.offroad_occurred or self.lane_invasion_occured:
            reward = -10
            done = True

        if self.render_mode:
            self.render()

        return observation, reward, done, False, info

    def _calculate_reward(self):
        """
        angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

        std = np.std(env.distance_from_center_history)
        distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)
        reward =  centering_factor + angle_factor + distance_std_factor
        """


        # Get vehicle transform and lane information
        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if waypoint is None:
            print("Offroad detected")
            self.offroad_occurred = True
            return -10  # Big penalty for going off-road

        lane_center = waypoint.transform.location
        distance_from_center = abs(location.y - lane_center.y)  # Assuming y is lateral direction

        # Reward for staying in lane center
        reward = max(0, 1.0 - (distance_from_center / 2.0))  # Normalize to 0-1 range
        return reward

    def render(self, mode='human'):
        if self.image is not None:
            cv2.imshow("CARLA Camera", self.image)
            cv2.waitKey(1)

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        cv2.destroyAllWindows()
