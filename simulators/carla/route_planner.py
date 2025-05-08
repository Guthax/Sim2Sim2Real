import math

from enum import Enum
from collections import deque
import random
import numpy as np
import carla

from simulators.carla.misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle

class RoutePlanner:
    def __init__(self, vehicle):
        world = vehicle.get_world()
        self._map = world.get_map()


        self.waypoints = []
        self.locations = []

        self.waypoints , self.locations = self.get_straight_waypoint_chain(vehicle.get_location())
        #print(f"Vehicle in  routeplanner: {vehicle.get_transform()}")

    def get_straight_waypoint_chain(self,start_location, distance=1.0, max_length=1000):


        current_wp = self._map.get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        start_location = current_wp.transform.location
        waypoints = [current_wp]
        locations = [current_wp.transform.location]
        while len(waypoints) < max_length:
            try:
                new_waypoints = current_wp.next_until_lane_end(distance)
                if len(new_waypoints) <= 1:
                    raise
                waypoints += new_waypoints
                locations += [wp.transform.location for wp in new_waypoints]
            except:
                next_wp = current_wp.next(distance + 3)
                if len(next_wp) > 1:
                    current_fwd = current_wp.transform.get_forward_vector()
                    best_wp = min(
                        next_wp,
                        key=lambda wp: abs(self.angle_between(current_fwd, wp.transform.get_forward_vector()))
                    )
                    next_wp = [best_wp]
                waypoints += next_wp
                locations += [wp.transform.location for wp in next_wp]
            current_wp = waypoints[-1]
        #print(f"First wpt in routeplanner: {waypoints[0]} for vehicle {start_location}")
        return waypoints, locations


    def angle_between(self, v1, v2):
        """Return the angle (in radians) between vectors v1 and v2."""
        dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        mag1 = math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2)
        mag2 = math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2)
        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)  # Clamp to avoid rounding issues
        return math.acos(cos_angle)

def is_waypoint_behind_vehicle(vehicle_transform, waypoint_transform):
    """
    Returns True if the waypoint is behind the vehicle.
    """
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()

    waypoint_location = waypoint_transform.location
    direction_to_waypoint = waypoint_location - vehicle_location

    dot_product = vehicle_forward.x * direction_to_waypoint.x + \
                  vehicle_forward.y * direction_to_waypoint.y + \
                  vehicle_forward.z * direction_to_waypoint.z

    return dot_product < 0
