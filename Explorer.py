#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading as thread

import matplotlib
import matplotlib.pyplot as plt

from hexapod_explorer.HexapodExplorer import HexapodExplorer
from hexapod_robot.HexapodRobot import HexapodRobot
from hexapod_robot.HexapodRobotConst import ROBOT_SIZE

sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')

from messages import *


class Explorer:
    """ Class to represent an exploration agent
    """

    def __init__(self, robotID=0):

        """ VARIABLES
        """
        # occupancy grid map of the robot ... possibly extended initialization needed in case of 'm1' assignment
        gridmap = OccupancyGrid()
        gridmap.resolution = 0.1
        gridmap.width = 100
        gridmap.height = 100
        gridmap.origin = Pose(Vector3(-5.0, -5.0, 0.0), Quaternion(1, 0, 0, 0))
        gridmap.data = 0.5 * np.ones(gridmap.height * gridmap.width)
        self.gridmap = gridmap

        # current frontiers
        self.frontiers = None

        # current path
        self.path = None

        # stopping condition
        self.stop = False

        """Connecting the simulator
        """
        # instantiate the robot
        self.robot = HexapodRobot(robotID)
        # ...and the explorer used in task t1c-t1e
        self.explor = HexapodExplorer()

    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        # turn on the robot
        self.robot.turn_on()

        # start navigation thread
        self.robot.start_navigation()

        # start the mapping thread
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start()
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)

        # start planning thread
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start()
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)

        # start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start()
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)

    def __del__(self):
        # turn off the robot
        self.robot.stop_navigation()
        self.robot.turn_off()

    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        while not self.stop:
            # fuse the laser scan
            laser_scan = self.robot.laser_scan_
            odometry = self.robot.odometry_
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)
            # ...

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path
        """
        while not self.stop:
            # obstacle growing
            gridmap_processed = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE)
            self.gridmap_processed = gridmap_processed
            # ...

            # frontier calculation
            frontiers = self.explor.find_free_edge_frontiers(self.gridmap)
            # ...

            if len(frontiers) == 0:
                print("frontiers empty")
                continue

            # path planning and goal selection
            odometry = self.robot.odometry_
            start = odometry.pose
            goal = frontiers[0]
            path = self.explor.plan_path(gridmap_processed, start, goal)
            path_simple = self.explor.simplify_path(gridmap_processed, path)
            # ...
            self.path = path_simple

            if self.path is not None:
                print("path set!")

    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """
        while not self.stop:
            # ...
            if self.path is None:
                # print("path none")
                continue

            if self.robot.navigation_goal is None:
                # fetch the new navigation goal
                path_nav = self.path.poses
                nav_goal = path_nav.pop(0)
                # give it to the robot
                self.robot.goto(nav_goal)
                print("Goto:" + nav_goal)
            # ...


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    # instantiate the robot
    ex0 = Explorer()
    # start the locomotion
    ex0.start()

    # continuously plot the map, targets and plan (once per second)
    fig, ax = plt.subplots()
    plt.ion()
    while (1):
        plt.cla()
        # plot the gridmap
        if ex0.gridmap_processed is not None and ex0.gridmap_processed.data is not None:
            ex0.gridmap_processed.plot(ax)
        # plot the navigation path
        if ex0.path is not None:
            ex0.path.plot(ax)

        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
        # to throttle the plotting pause for 1s
        plt.pause(1)
