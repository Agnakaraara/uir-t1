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

    frontiers: [Pose] = []
    path: Path = None
    currentWaypointIndex: int
    robot: HexapodRobot
    explor = HexapodExplorer()

    def __init__(self, robotID=0):
        self.robot = HexapodRobot(robotID)

    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning """

        self.robot.turn_on()
        self.robot.start_navigation()

        planning_thread = Thread(target=self.planning)
        planning_thread.start()

        traj_follow_thread = thread.Thread(target=self.trajectory_following)
        traj_follow_thread.start()

    def __del__(self):
        self.robot.stop_navigation()
        self.robot.turn_off()

    def planning(self, master):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path """
        while not master.stop:
            time.sleep(0.5)
            odometry = self.robot.odometry_
            gridMapP = copy.deepcopy(master.gridMapP)
            if gridMapP is None or self.path is not None and self.explor.isPathTraversable([odometry.pose] + self.path.poses[self.currentWaypointIndex:], gridMapP) and self.robot.navigation_goal is not None: continue

            start = odometry.pose
            goal: Pose

            if sys.argv[1] == "p1":
                self.frontiers = self.explor.find_free_edge_frontiers(master.gridMap, gridMapP, odometry)
                goal = self.explor.pick_frontier_closest(self.frontiers, gridMapP, odometry)
            elif sys.argv[1] == "p2":
                frontiers = self.explor.find_inf_frontiers(master.gridMap, gridMapP, odometry)
                self.frontiers = list(map(lambda x: x[0], frontiers))
                goal = self.explor.pick_frontier_inf(frontiers, gridMapP, odometry)
            elif sys.argv[1] == "p3":
                self.frontiers = self.explor.find_free_edge_frontiers(master.gridMap, gridMapP, odometry)
                goal = self.explor.pick_frontier_tsp(self.frontiers, gridMapP, odometry)

            if len(self.frontiers) == 0:
                master.stop = True
                print("No more frontiers! Stopping robot.")
                return

            pathRaw = self.explor.plan_path(gridMapP, start, goal)
            pathSimple = self.explor.simplify_path(gridMapP, pathRaw)
            self.path = pathSimple
            self.currentWaypointIndex = -1
            print("Path recalculated.")

    def trajectory_following(self):
        """ trajectory following thread that assigns new goals to the robot navigation thread """
        while not master.stop:
            time.sleep(0.5)
            if self.path is None or self.currentWaypointIndex == len(self.path.poses)-1: continue
            if self.robot.navigation_goal is None or self.currentWaypointIndex == -1:   # if robot is not already going somewhere
                self.currentWaypointIndex += 1
                waypoint = self.path.poses[self.currentWaypointIndex]
                self.robot.goto(waypoint)
                print("Goto:", waypoint.position.x, waypoint.position.y)


class Master:

    gridMap: OccupancyGrid = None
    gridMapP: OccupancyGrid = None
    stop = False
    explorers: [Explorer]
    explor = HexapodExplorer()

    def __init__(self):
        gridMap = OccupancyGrid()
        gridMap.resolution = 0.1
        gridMap.width = 1
        gridMap.height = 1
        gridMap.data = 0.5 * np.ones(gridMap.height * gridMap.width)
        self.gridMap = gridMap

    def start(self):

        mapping_thread = Thread(target=self.mapping)
        mapping_thread.start()

        for ex in self.explorers:
            ex.start()

    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map """
        while not self.stop:
            time.sleep(0.5)
            for ex in self.explorers:
                laser_scan = ex.robot.laser_scan_
                odometry = ex.robot.odometry_
                gridMap = self.explor.fuse_laser_scan(self.gridMap, laser_scan, odometry)
                gridMapP = self.explor.grow_obstacles(gridMap, ROBOT_SIZE)
                self.gridMap = gridMap
                self.gridMapP = gridMapP


if __name__ == "__main__":
    matplotlib.use('TkAgg')

    master = Master()
    master.explorers = [Explorer(0), Explorer(1)]
    master.start()

    fig, axis = plt.subplots()
    plt.ion()
    while True:
        plt.cla()   # clear axis
        if master.gridMap is not None:
            master.gridMap.plot(axis)
        for ex in master.explorers:
            if ex.robot.odometry_ is not None:
                plt.plot([ex.robot.odometry_.pose.position.x], [ex.robot.odometry_.pose.position.y], "gD")
            for frontier in ex.frontiers:
                plt.plot([frontier.position.x], [frontier.position.y], 'o')
            if ex.path is not None:
                ex.path.plot(axis)
            if ex.robot.navigation_goal is not None:
                plt.plot([ex.robot.navigation_goal.position.x], [ex.robot.navigation_goal.position.y], 'x', markersize=10)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        axis.set_aspect('equal', 'box')
        plt.show()
        plt.pause(0.5)
