#!/usr/bin/env python

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


class RoutePlanner():
    def __init__(self, vehicle, buffer_size, ignore_intersections=True):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = 1
        self._min_distance = 1

        self._ignore_intersections = ignore_intersections

        self._target_waypoint = None
        self._buffer_size = buffer_size
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW

        self._last_traffic_light = None
        self._proximity_threshold = 15.0


        self._compute_next_waypoints(k=300)

    def _compute_next_waypoints(self, k=1):
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                road_options_list = retrieve_options(next_waypoints, last_waypoint)
                next_waypoint = next_waypoints[0]
                road_option = road_options_list[0]

            else:
                road_options_list = retrieve_options(next_waypoints, last_waypoint)

                # Try to pick a STRAIGHT path first
                preferred_options = [RoadOption.STRAIGHT]
                straight_indices = [i for i, option in enumerate(road_options_list)
                                    if option in preferred_options]

                if self._ignore_intersections and straight_indices:
                    selected_index = straight_indices[0]
                else:
                    selected_index = random.randint(0, len(next_waypoints) - 1)

                next_waypoint = next_waypoints[selected_index]
                road_option = road_options_list[selected_index]

            #print(f"Junction: {next_waypoint.is_junction}, Selected option: {road_option}")
            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self):
        waypoints = self._get_waypoints()
        # red_light = False
        return waypoints

    def _get_waypoints(self):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        #   Buffering the waypoints
        while len(self._waypoint_buffer) < self._buffer_size:
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        waypoints = []

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            waypoints.append(waypoint)

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index - 1):
                self._waypoint_buffer.popleft()

        return waypoints

def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
         candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
         RoadOption.STRAIGHT
         RoadOption.LEFT
         RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT