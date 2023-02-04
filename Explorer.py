#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading as thread
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt

from hexapod_explorer.HexapodExplorer import HexapodExplorer
from hexapod_robot.HexapodRobot import HexapodRobot
from hexapod_robot.HexapodRobotConst import ROBOT_SIZE
from messages import *

sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')


class Explorer:

    gridMap: OccupancyGrid
    gridMapP: OccupancyGrid
    frontiers: [Pose] = []
    path: Path = None
    stop = False
    robot: HexapodRobot
    explor = HexapodExplorer()

    def __init__(self, robotID=0):

        gridMap = OccupancyGrid()
        gridMap.resolution = 0.1
        gridMap.width = 100
        gridMap.height = 100
        gridMap.origin = Pose(Vector3(-5.0, -5.0, 0.0), Quaternion(1, 0, 0, 0))
        gridMap.data = 0.5 * np.ones(gridMap.height * gridMap.width)
        self.gridMap = gridMap

        self.robot = HexapodRobot(robotID)

    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning """

        self.robot.turn_on()
        self.robot.start_navigation()

        mapping_thread = Thread(target=self.mapping)
        mapping_thread.start()

        planning_thread = Thread(target=self.planning)
        planning_thread.start()

        traj_follow_thread = thread.Thread(target=self.trajectory_following)
        traj_follow_thread.start()

    def __del__(self):
        self.robot.stop_navigation()
        self.robot.turn_off()

    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map """
        while not self.stop:
            laser_scan = self.robot.laser_scan_
            odometry = self.robot.odometry_
            self.gridMap = self.explor.fuse_laser_scan(self.gridMap, laser_scan, odometry)

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path """
        while not self.stop:
            self.gridMapP = self.explor.grow_obstacles(self.gridMap, ROBOT_SIZE)
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridMap, self.gridMapP)

            if len(self.frontiers) == 0:
                print("frontiers empty")
                continue

            start = self.robot.odometry_.pose
            goal = self.explor.pick_frontier_closest(self.frontiers, self.gridMapP, self.robot.odometry_)
            pathRaw = self.explor.plan_path(self.gridMapP, start, goal)
            pathSimple = self.explor.simplify_path(self.gridMapP, pathRaw)
            self.path = pathSimple

            if self.path is not None:
                print("path set!")

    def trajectory_following(self):
        """ trajectory following thread that assigns new goals to the robot navigation thread """
        while not self.stop:

            if self.path is None:
                # print("path none")
                continue

            if self.robot.navigation_goal is None:      # if robot is not already going somewhere
                waypoint = self.path.poses.pop(0)
                self.robot.goto(waypoint)
                print("Goto:", waypoint)


if __name__ == "__main__":
    matplotlib.use('TkAgg')

    ex0 = Explorer()
    ex0.start()

    fig, axis = plt.subplots()
    plt.ion()
    while True:
        plt.cla()   # clear axis
        if ex0.gridMapP is not None and ex0.gridMapP.data is not None:
            ex0.gridMapP.plot(axis)
        for frontier in ex0.frontiers:
            plt.plot([frontier.position.x], [frontier.position.y], '.', markersize=20)
        if ex0.path is not None:
            ex0.path.plot(axis)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        axis.set_aspect('equal', 'box')
        plt.show()
        plt.pause(1)
