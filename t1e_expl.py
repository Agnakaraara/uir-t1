#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt

from hexapod_explorer.HexapodExplorer import HexapodExplorer

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
            #find free edges
            points = explor.find_inf_frontiers(gridmap)

            #plot the map
            fig, ax = plt.subplots()
            #plot the gridmap
            gridmap.plot(ax)
            #plot the cluster centers
            if points is not None:
                for p in points:
                    plt.plot([p.position.x],[p.position.y],'.', markersize=20)

            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            ax.set_aspect('equal', 'box')
            plt.show()
