#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt

from hexapod_explorer.HexapodExplorer import HexapodExplorer
from messages import Odometry, Pose

sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
 
#import hexapod robot and explorer

#import communication messages

#pickle
import pickle

#switch to select between the simple variant or the hard variant
SIMPLE_VARIANT = True

if __name__=="__main__":

    explor = HexapodExplorer()

    dataset = pickle.load( open( "resources/frontier_detection.bag", "rb" ) )

    for gridmap in dataset:

            gridMapP = explor.grow_obstacles(gridmap, 0.3)

            #find free edges
            points = explor.find_free_edge_frontiers(gridmap, gridMapP)

            start = Pose()
            for x in range(gridMapP.width):
                for y in range(gridMapP.height):
                    if gridMapP.data.reshape([gridmap.height, gridmap.width])[y, x] == 0:
                        start = explor.cellToPose((x, y), gridmap)

            odomentry = Odometry()
            odomentry.pose = start
            frontier = explor.pick_frontier_closest(points, gridMapP, odomentry)
            path = explor.plan_path(gridmap, start, frontier)
            path = explor.simplify_path(gridmap, path)

            #plot the map
            fig, ax = plt.subplots()
            #plot the gridmap
            gridMapP.plot(ax)
            plt.plot([start.position.x], [start.position.y], '.', markersize=20)
            path.plot(ax)

            #plot the cluster centers
            if points is not None:
                for p in points:
                    plt.plot([p.position.x],[p.position.y],'.', markersize=20)

            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            ax.set_aspect('equal', 'box')
            plt.show()
