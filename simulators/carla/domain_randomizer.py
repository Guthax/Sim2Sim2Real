import random
import carla

class CarlaRandomizer:
    def __init__(self, env, max_steps):
        self.env = env
        self.max_steps = max_steps
        self.current_step = 0

    def get_scale(self):
        return min(1.0, self.current_step / self.max_steps)

    def randomize_environment(self):
        scale = self.get_scale()

        # Randomize weather
        cloudiness = random.uniform(0.0, 80.0 * scale),
        precipitation = random.uniform(0.0, 30.0 * scale),  # Light rain only
        wind_intensity = random.uniform(0.0, 40.0 * scale),
        sun_altitude_angle = random.uniform(15.0, 75.0 * scale),  # Keep sun up
        fog_density = random.uniform(0.0, 60.0 * scale),  # Light fog
        fog_distance = random.uniform(20.0, 100.0 * scale),  # Allow visibility
        wetness = random.uniform(0.0, 70.0 * scale)


        weather = carla.WeatherParameters(
            cloudiness=cloudiness,
            precipitation=precipitation,
            wind_intensity =wind_intensity,
            fog_density=fog_density,
            fog_distance = fog_distance,
            wetness=wetness,
            sun_altitude_angle=sun_altitude_angle
        )
        self.env.world.set_weather(weather)

        # (Optional) Randomize road friction
        # you would modify friction on certain roads if you want

        # (Optional) Randomize vehicle physics
        # mass change, tire friction, center of mass, etc.

    def reset_step(self, step_increment):
        self.current_step += step_increment
