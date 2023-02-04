import time
from threading import Thread

import numpy as np

from Explorer import Explorer
from hexapod_explorer.HexapodExplorer import HexapodExplorer
from hexapod_robot.HexapodRobotConst import ROBOT_SIZE
from messages import OccupancyGrid


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
                