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
            points = explor.find_inf_frontiers(gridmap, gridMapP)

            #plot the map
            fig, ax = plt.subplots()
            #plot the gridmap
            gridmap.plot(ax)
           # plt.plot([start.position.x], [start.position.y], '.', markersize=20)
           # path.plot(ax)

            #plot the cluster centers
            if points is not None:
                for p in points:
                    plt.plot([p[0].position.x],[p[0].position.y],'.', markersize=20)

            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            ax.set_aspect('equal', 'box')
            plt.show()
