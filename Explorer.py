#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
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

    gridMap: OccupancyGrid = None
    gridMapP: OccupancyGrid = None
    frontiers: [Pose] = []
    path: Path = None
    currentWaypointIndex: int
    stop = False
    robot: HexapodRobot
    explor = HexapodExplorer()

    def __init__(self, robotID=0):

        gridMap = OccupancyGrid()
        gridMap.resolution = 0.1
        gridMap.width = 1
        gridMap.height = 1
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
            time.sleep(0.5)
            laser_scan = self.robot.laser_scan_
            odometry = self.robot.odometry_
            gridMap = self.explor.fuse_laser_scan(self.gridMap, laser_scan, odometry)
            gridMapP = self.explor.grow_obstacles(gridMap, ROBOT_SIZE)
            self.gridMap = gridMap
            self.gridMapP = gridMapP

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path """
        while not self.stop:
            time.sleep(0.5)
            gridMapP = copy.deepcopy(self.gridMapP)
            if gridMapP is None or self.path is not None and self.explor.isPathTraversable([self.robot.odometry_.pose] + self.path.poses[self.currentWaypointIndex:], gridMapP) and self.robot.navigation_goal is not None: continue
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridMap, gridMapP)
            if len(self.frontiers) == 0: continue

            start = self.robot.odometry_.pose
            goal = self.explor.pick_frontier_closest(self.frontiers, gridMapP, self.robot.odometry_)
            pathRaw = self.explor.plan_path(gridMapP, start, goal)
            pathSimple = self.explor.simplify_path(gridMapP, pathRaw)
            self.path = pathSimple
            self.currentWaypointIndex = -1

    def trajectory_following(self):
        """ trajectory following thread that assigns new goals to the robot navigation thread """
        while not self.stop:
            time.sleep(0.5)
            if self.path is None or self.currentWaypointIndex == len(self.path.poses)-1: continue
            if self.robot.navigation_goal is None or self.currentWaypointIndex == -1:   # if robot is not already going somewhere
                self.currentWaypointIndex += 1
                waypoint = self.path.poses[self.currentWaypointIndex]
                self.robot.goto(waypoint)
                print("Goto:", waypoint.position.x, waypoint.position.y)


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
        if ex0.robot.odometry_ is not None:
            plt.plot([ex0.robot.odometry_.pose.position.x], [ex0.robot.odometry_.pose.position.y], "D")
        for frontier in ex0.frontiers:
            plt.plot([frontier.position.x], [frontier.position.y], 'o')
        if ex0.path is not None:
            ex0.path.plot(axis)
        if ex0.robot.navigation_goal is not None:
            plt.plot([ex0.robot.navigation_goal.position.x], [ex0.robot.navigation_goal.position.y], 'X')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        axis.set_aspect('equal', 'box')
        plt.show()
        plt.pause(0.5)
