#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import copy

#cpg network
import cpg.oscilator_network as osc

#import messages
from messages import *

import matplotlib.pyplot as plt

import scipy.ndimage as ndimg
from sklearn.cluster import KMeans

import skimage.measure

import collections
import heapq

class HexapodExplorer:

    def __init__(self):
        pass

    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)

        #TODO:[t1c_map] fuse the correctly aligned laser data into the probabilistic occupancy grid map

        P: [float] = laser_scan.distances
        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        robot_position = odometry.pose.position[:2]  # just x,y
        robot_angle = odometry.pose.orientation.to_Euler()[0]
        grid_origin = grid_map.origin
        grid_resolution = grid_map.resolution
        grid_width = grid_map.width
        grid_height = grid_map.height

        c, s = np.cos(robot_angle), np.sin(robot_angle)
        R = np.array(((c, -s), (s, c)))

        for i in P:
            P[i] = np.array([math.cos(angle_min + i*angle_increment) * P[i], math.sin(angle_min + i*angle_increment) * P[i]])
            P[i] = R*P[i] + robot_position
            P[i] = self.world_to_map(P[i], grid_origin, grid_resolution)
            P[i][0] = np.max(0, np.min(grid_width-1, P[i][0]))
            P[i][1] = np.max(0, np.min(grid_height-1, P[i][1]))

        odom_map = self.world_to_map(odometry.position, grid_origin, grid_resolution)
        laser_scan_points_map = P

        free_points = []
        occupied_points = []

        for pt in laser_scan_points_map:
            pts = self.bresenham_line(odom_map, pt)
            free_points.extend(pts)
            occupied_points.append(pt)

        # Bayesian update

        data = grid_map.data.reshape(grid_map_update.height, grid_map_update.width)

        for free_point in free_points:
            data[free_point[0], free_point[1]] = self.update_free(data[free_point[0], free_point[1]])

        for occupied_point in occupied_points:
            data[occupied_point[0], occupied_point[1]] = self.update_occupied(data[occupied_point[0], occupied_point[1]])

        #serialize the data back (!watch for the correct width and height settings if you are doing the harder assignment)
        grid_map_update.data = data.flatten()

        return grid_map_update


    def world_to_map(self, p, grid_origin, grid_resolution):
        return np.round((p - grid_origin) / grid_resolution)

    def update_free(self, P_mi):
        """method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """

        S_z_occupied = 0
        S_z_free = 0.95
        p_z_mi_occupied = (1 + S_z_occupied - S_z_free)/2
        p_mi_occupied = P_mi   # previous value
        p_z_mi_free = 1-p_z_mi_occupied
        p_mi_free = 1-p_mi_occupied

        p_mi = (p_z_mi_occupied * p_mi_occupied)/(p_z_mi_occupied*p_mi_occupied + p_z_mi_free*p_mi_free)

        if p_mi == 0:
            p_mi = 0.01

        return p_mi

    def update_occupied(self, P_mi):
        """method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        S_z_occupied = 0.95
        S_z_free = 0
        p_z_mi_occupied = (1 + S_z_occupied - S_z_free)/2
        p_mi_occupied = P_mi   # previous value
        p_z_mi_free = 1-p_z_mi_occupied
        p_mi_free = 1-p_mi_occupied

        p_mi = (p_z_mi_occupied * p_mi_occupied)/(p_z_mi_occupied*p_mi_occupied + p_z_mi_free*p_mi_free)

        if p_mi == 0:
            p_mi = 0.01

        return p_mi

    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find free-adges and cluster the frontiers
        return None 


    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find the information rich points in the environment
        return None


    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """

        grid_map_grow = copy.deepcopy(grid_map)

        #TODO:[t1d-plan] grow the obstacles for robot_size

        return grid_map_grow


    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        path = Path()
        #add the start pose
        path.poses.append(start)
        
        #TODO:[t1d-plan] plan the path between the start and the goal Pose
        
        #add the goal pose
        path.poses.append(goal)

        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map == None or path == None:
            return None
 
        path_simplified = Path()
        #add the start pose
        path_simplified.poses.append(path.poses[0])
        
        #TODO:[t1d-plan] simplifie the planned path
        
        #add the goal pose
        path_simplified.poses.append(path.poses[-1])

        return path_simplified
 
    ###########################################################################
    #INCREMENTAL Planner
    ###########################################################################

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if not hasattr(self, 'rhs'): #first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)

        #TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()

    # Support stuff

    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x,y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x,y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

