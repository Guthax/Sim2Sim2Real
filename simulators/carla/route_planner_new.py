#!/usr/bin/env python
import math
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from enum import Enum
from collections import deque
import random
import numpy as np
import carla

from simulators.carla.misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


class RoutePlannerNew():
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        vehicle_location = self._vehicle.get_location()
        start_location = self._map.get_waypoint(vehicle_location, project_to_road=False, lane_type=carla.LaneType.Driving).get_right_lane().transform.location
        # Generate straight path waypoints
        waypoints = self.get_straight_waypoint_chain(start_location)

        # Optionally, draw them in the simulation
        for wp in waypoints:

            self._world.debug.draw_point(
                carla.Location(wp.transform.location.x, wp.transform.location.y, 0.25), 0.1,
                carla.Color(255, 0, 0),
                30, False)

    def get_straight_waypoint_chain(self,start_location, distance=1.0, junction_distance=1.0, max_length=1000,
                                    max_angle_deg=10.0 , loop_tolerance=2.0):


        current_wp = self._map.get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        start_location = current_wp.transform.location
        waypoints = [current_wp]
        while len(waypoints) < max_length:
            try:
                new_waypoints = current_wp.next_until_lane_end(distance)
                if len(new_waypoints) <= 1:
                    raise
                waypoints += new_waypoints
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
            current_wp = waypoints[-1]
        return waypoints


    def angle_between(self, v1, v2):
        """Return the angle (in radians) between vectors v1 and v2."""
        dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        mag1 = math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2)
        mag2 = math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2)
        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)  # Clamp to avoid rounding issues
        return math.acos(cos_angle)