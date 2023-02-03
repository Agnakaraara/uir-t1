#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import heapq

# import messages
import scipy.ndimage
import scipy.ndimage as ndimg
import skimage.measure
from sklearn.cluster import KMeans

from hexapod_explorer.a_star import a_star
from hexapod_explorer.gridmap import OccupancyGridMap
from hexapod_robot.HexapodRobotConst import LASER_SCAN_RANGE_MAX, LASER_SCAN_RANGE_MIN
from lkh.invoke_LKH import solve_TSP
from messages import *


# cpg network


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        return heapq.heappop(self.elements)[1]

    def top(self):
        u = self.elements[0]
        return u[1]

    def topKey(self):
        u = self.elements[0]
        return u[0]

    def contains(self, element):
        ret = False
        for item in self.elements:
            if element == item[1]:
                ret = True
                break
        return ret

    def print_elements(self):
        print(self.elements)
        return self.elements

    def remove(self, element):
        i = 0
        for item in self.elements:
            if element == item[1]:
                self.elements[i] = self.elements[-1]
                self.elements.pop()
                heapq.heapify(self.elements)
                break
            i += 1


class HexapodExplorer:

    # t1c - Map, project M1 + M2

    def fuse_laser_scan(self, grid_map: OccupancyGrid, laser_scan: LaserScan, odometry: Odometry) -> OccupancyGrid:
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

        if laser_scan is None or odometry is None:
            return grid_map_update

        P: [float] = laser_scan.distances
        robot_position = np.array([odometry.pose.position.x, odometry.pose.position.y])  # just x,y
        robot_rotation_matrix = odometry.pose.orientation.to_R()[0:2, 0:2]
        grid_origin = np.array([grid_map.origin.position.x, grid_map.origin.position.y])

        for i in range(0, len(P)):
            if P[i] < laser_scan.range_min: P[i] = laser_scan.range_min
            if P[i] > laser_scan.range_max: P[i] = laser_scan.range_max
            angle = laser_scan.angle_min + i * laser_scan.angle_increment
            P[i] = np.array([math.cos(angle) * P[i], math.sin(angle) * P[i]])
            P[i] = robot_rotation_matrix @ np.transpose(P[i]) + robot_position
            P[i] = self.world_to_map(P[i], grid_origin, grid_map.resolution)
            P[i][0] = max(0, min(grid_map.width - 1, P[i][0]))
            P[i][1] = max(0, min(grid_map.height - 1, P[i][1]))

        odom_map = self.world_to_map(robot_position, grid_origin, grid_map.resolution)
        laser_scan_points_map = P

        free_points = []
        occupied_points = []

        for pt in laser_scan_points_map:
            pts = self.bresenham_line(odom_map, pt)
            free_points.extend(pts)
            occupied_points.append(pt)

        # Bayesian update

        data = grid_map.data.reshape(grid_map_update.height, grid_map_update.width)

        for (x, y) in free_points:
            data[y, x] = self.update_free(data[y, x])

        for (x, y) in occupied_points:
            data[y, x] = self.update_occupied(data[y, x])

        # serialize the data back (!watch for the correct width and height settings if you are doing the harder assignment)
        grid_map_update.data = data.flatten()

        return grid_map_update

    def update_free(self, P_mi):
        """method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """

        S_z_occupied = 0
        S_z_free = 0.95
        p_z_mi_occupied = (1 + S_z_occupied - S_z_free) / 2
        p_mi_occupied = P_mi  # previous value
        p_z_mi_free = 1 - p_z_mi_occupied
        p_mi_free = 1 - p_mi_occupied

        p_mi = (p_z_mi_occupied * p_mi_occupied) / (p_z_mi_occupied * p_mi_occupied + p_z_mi_free * p_mi_free)

        return max(0.05, p_mi)  # never let p_mi get to 0

    def update_occupied(self, P_mi):
        """method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        S_z_occupied = 0.95
        S_z_free = 0
        p_z_mi_occupied = (1 + S_z_occupied - S_z_free) / 2
        p_mi_occupied = P_mi  # previous value
        p_z_mi_free = 1 - p_z_mi_occupied
        p_mi_free = 1 - p_mi_occupied

        p_mi = (p_z_mi_occupied * p_mi_occupied) / (p_z_mi_occupied * p_mi_occupied + p_z_mi_free * p_mi_free)

        return min(p_mi, 0.95)  # never let p_mi get to 1

    # t1e - Frontiers

    # F1 + F2

    def find_free_edge_frontiers(self, grid_map: OccupancyGrid, gridMapP: OccupancyGrid) -> [Pose]:
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        # free-edge cell detection

        data = grid_map.data.copy().reshape(grid_map.height, grid_map.width)
        dataP = gridMapP.data.reshape(grid_map.height, grid_map.width)

        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                if data[x, y] == 0.5:
                    data[x, y] = 10

        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        data_c = ndimg.convolve(data, mask, mode='constant', cval=0.0)

        for x in range(0, data_c.shape[1]):
            for y in range(0, data_c.shape[0]):
                if data[y, x] < 0.5 and 10 <= data_c[y, x] < 50 and dataP[y, x] == 0:    # cell is free-edge
                    data[y, x] = 1
                else:
                    data[y, x] = 0

        # free-edge clustering

        labeled_image, num_labels = skimage.measure.label(data, connectivity=2, return_num=True)

        clusters = {}

        for label in range(1, num_labels+1):
            clusters[label] = []

        for x in range(0, labeled_image.shape[1]):
            for y in range(0, labeled_image.shape[0]):
                label = labeled_image[y, x]
                if label != 0:
                    clusters[label].append((x, y))

        # free-edge centroids

        pose_list = []

        for label, cells in clusters.items():
            f = len(cells)
            D = LASER_SCAN_RANGE_MAX / grid_map.resolution
            n_r = int(1 + np.floor(f/D + 0.5))
            kmeans = KMeans(n_clusters=n_r, random_state=0, n_init="auto").fit(cells)
            for centroid in kmeans.cluster_centers_:
                if dataP[int(centroid[1]), int(centroid[0])] == 0:    # ignore unreachable frontiers
                    pose_list.append(self.cellToPose(centroid, grid_map))

            # t1e
            #centroid = (0, 0)
            #for cell in cells:
            #    centroid = (centroid[0] + cell[0], centroid[1] + cell[1])
            #centroid = (centroid[0] / len(cells), centroid[1] / len(cells))

            #pose = Pose()
            #pose.position.x = centroid[0] * grid_map.resolution + grid_map.origin.position.x
            #pose.position.y = centroid[1] * grid_map.resolution + grid_map.origin.position.y
            #pose_list.append(pose)

        return pose_list

    # F3

    def find_inf_frontiers(self, grid_map: OccupancyGrid) -> [(Pose, float)]:  # project F3
        """Method to find the frontiers based on information theory approach"""

        frontiersWeighted = []

        H = grid_map.data.copy()
        for x in range(len(H)):
            p = H[x]
            H[x] = 0 if p == 0 or p == 1 else -p * np.log(p) - (1-p) * np.log(1-p)
        H = H.reshape((grid_map.height, grid_map.width))

        frontiers = self.find_free_edge_frontiers(grid_map)

        range_max = LASER_SCAN_RANGE_MAX / grid_map.resolution
        range_min = LASER_SCAN_RANGE_MIN / grid_map.resolution

        for frontier in frontiers:
            frontier_cell = self.poseToCell(frontier, grid_map)
            I_action = 0.0
            for y in range(max(int(frontier_cell[1]-range_max), 0), min(int(frontier_cell[1]+range_max+1), grid_map.height)):
                for x in range(max(int(frontier_cell[0]-range_max), 0), min(int(frontier_cell[0]+range_max+1), grid_map.width)):
                    dist = self.distanceOfCellsEuclidean((x, y), frontier_cell)
                    if range_min <= dist <= range_max and self.cellsSeeEachOther((x, y), frontier_cell, grid_map):
                        I_action += H[y, x]
            frontiersWeighted.append((frontier, I_action))

        frontiersWeighted.sort(key=lambda wf: -wf[1])

        return frontiersWeighted

    # Project - Pick frontier

    # P1

    def pick_frontier_closest(self, frontiers: [Pose], gridMapP: OccupancyGrid, odometry: Odometry) -> Pose:
        closest = None
        minDist = np.inf
        for frontier in frontiers:
            dist = self.distanceOfPosesAStar(frontier, odometry.pose, gridMapP)
            if dist < minDist:
                minDist = dist
                closest = frontier
        return closest

    # P2

    def pick_frontier_inf(self, frontiers: [Pose, float], gridMapP: OccupancyGrid, odometry: Odometry) -> Pose:
        best_frontiers = []
        maxUtility = -np.inf
        for frontier, utility in frontiers:     # utility = mutual information computed in F3
            if utility > maxUtility:
                maxUtility = utility
                best_frontiers = [frontier]
            elif utility == maxUtility:
                best_frontiers.append(frontier)
        return self.pick_frontier_closest(best_frontiers, gridMapP, odometry)

    # P3

    def pick_frontier_tsp(self, frontiers: [Pose], gridMapP: OccupancyGrid, odometry: Odometry) -> Pose:
        if len(frontiers) == 1:
            return frontiers[0]
        points = [odometry.pose] + frontiers
        n = len(points)
        distance_matrix = np.zeros((n, n))
        for a in range(n):         # from
            for b in range(1, n):     # to, because it's open-ended we keep first one zero
                f1 = points[a]
                f2 = points[b]
                distance_matrix[a][b] = self.distanceOfPosesAStar(f1, f2, gridMapP)
        sequence = solve_TSP(distance_matrix)
        return frontiers[sequence[1]-1]

    # t1d - Plan path, A-star

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """

        grid_map_grow = copy.deepcopy(grid_map)

        # TODO:[t1d-plan] grow the obstacles for robot_size

        data = grid_map.data.copy().reshape(grid_map_grow.height, grid_map_grow.width)

        ranges = data.copy()

        for y in range(0, grid_map_grow.height):
            for x in range(0, grid_map_grow.width):
                p = data[y, x]
                if p > 0.5:
                    ranges[y, x] = 0
                elif p == 0.5:
                    ranges[y, x] = 1    # Set to 1 so unknown places will not grow as they are not obstacles
                else:
                    ranges[y, x] = 1

        ranges = scipy.ndimage.distance_transform_edt(ranges)

        for y in range(0, grid_map_grow.height):
            for x in range(0, grid_map_grow.width):
                p = data[y, x]
                if p > 0.5:
                    data[y, x] = 1
                elif p == 0.5:
                    data[y, x] = 1
                elif p < 0.5 and ranges[y, x] * grid_map_grow.resolution < robot_size:
                    data[y, x] = 1
                else:
                    data[y, x] = 0

        grid_map_grow.data = data.flatten()

        return grid_map_grow

    def plan_path(self, gridMapP: OccupancyGrid, start: Pose, goal: Pose):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            gridMapP: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        path = Path()
        # add the start pose
        # path.poses.append(start)

        # TODO:[t1d-plan] plan the path between the start and the goal Pose

        data = gridMapP.data.reshape(gridMapP.height, gridMapP.width)

        gmap = OccupancyGridMap(data, gridMapP.resolution)

        try:
            result = a_star((start.position.x, start.position.y), (goal.position.x, goal.position.y), gmap)
        except:
            return None            # start or end are blocked

        for path_point in result[0]:
            pose = Pose()
            pose.position.x = path_point[0]
            pose.position.y = path_point[1]
            path.poses.append(pose)

        if len(path.poses) == 0:
            return None            # there is an obstacle blocking the path

        return path

    def simplify_path(self, gridMapP, path: Path) -> Path:
        """ Method to simplify the found path on the grid
        Args:
            gridMapP: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if gridMapP is None or path is None:
            return None

        grid_origin = np.array([gridMapP.origin.position.x, gridMapP.origin.position.y])

        path_simplified = Path()
        # add the start pose
        path_simplified.poses.append(path.poses[0])

        # TODO:[t1d-plan] simplifie the planned path

        data = gridMapP.data.reshape(gridMapP.height, gridMapP.width)

        i = 1

        # iterate through the path and simplify the path
        while path_simplified.poses[-1] != path.poses[-1]:  # until the goal is not reached
            # find the connected segment
            previous_pose = path_simplified.poses[-1]
            for pose in path.poses[i:]:
                end = path_simplified.poses[-1]
                end = self.world_to_map(np.array([end.position.x, end.position.y]), grid_origin, gridMapP.resolution)
                pose_point = self.world_to_map(np.array([pose.position.x, pose.position.y]), grid_origin,
                                               gridMapP.resolution)
                line = self.bresenham_line(end, pose_point)
                collide = False
                for point in line:
                    if data[point[1], point[0]] == 1:
                        collide = True

                if not collide:  # there is no collision
                    previous_pose = pose
                    i += 1

                    # the goal is reached
                    if pose == path.poses[-1]:
                        path_simplified.poses.append(pose)
                        break

                else:  # there is collision
                    path_simplified.poses.append(previous_pose)
                    break

        # add the goal pose
        # path_simplified.poses.append(path.poses[-1])

        return path_simplified

    ###########################################################################
    # INCREMENTAL Planner
    ###########################################################################

    # t1x - D-star

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        grid_origin = np.array([grid_map.origin.position.x, grid_map.origin.position.y])
        grid_resolution = grid_map.resolution

        start = tuple(self.world_to_map(np.array([start.position.x, start.position.y]), grid_origin, grid_resolution))
        goal = tuple(self.world_to_map(np.array([goal.position.x, goal.position.y]), grid_origin, grid_resolution))

        if not hasattr(self, 'rhs'):  # first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)
            self.gridmap = copy.deepcopy(grid_map.data.reshape(grid_map.height, grid_map.width)).transpose()
            self.initialize(goal)

        data = grid_map.data.reshape(grid_map.height, grid_map.width).transpose()

        changed = False

        for x in range(0, grid_map.width):
            for y in range(0, grid_map.height):
                if data[x, y] != self.gridmap[x, y]:
                    changed = True
                    self.gridmap[x, y] = data[x, y]
                    for s in self.neighbors8([x, y]):
                        self.update_vertex(s, start, goal)

        if changed:
            for element in self.U.print_elements():
                self.U.remove(element[1])
                self.U.put(element[1], self.calculate_key(element[1], goal))

        self.compute_shortest_path(start, goal)

        if self.g[start] == np.inf:
            path = None
        else:
            path = self.reconstruct_path(start, goal)

        return path, self.rhs, self.g

    gridmap = None
    U: PriorityQueue = None

    def calculate_key(self, coord, goal):
        """method to calculate the priority queue key
        Args:  coord: (int, int) - cell to calculate key for
               goal: (int, int) - goal location
        Returns:  (float, float) - major and minor key
        """
        # think about different heuristics and how they influence the steering of the algorithm
        # heuristics = 0
        # heuristics = L1 distance
        # heuristics = L2 distance

        return [min(self.g[coord], self.rhs[coord]), min(self.g[coord], self.rhs[coord])]

    def initialize(self, goal):
        self.U = PriorityQueue()
        self.rhs[goal] = 0
        self.U.put(goal, self.calculate_key(goal, goal))

    def update_vertex(self, u, start, goal):
        """ Function for map vertex updating
        Args:  u: (int, int) - currently processed position
               start: (int, int) - start position
               goal: (int, int) - goal position
        """
        if u != goal:
            mn = np.inf
            for s in self.neighbors8(u):
                mn = min(mn, self.edge_cost(u, s) + self.g[s])
            self.rhs[u] = mn
        self.U.remove(u)
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.calculate_key(u, goal))

    def edge_cost(self, uFrom, uTo):
        if self.gridmap[uTo] == 1:
            return np.inf
        else:
            return np.sqrt((uFrom[0] - uTo[0]) ** 2 + (uFrom[1] - uTo[1]) ** 2)

    def compute_shortest_path(self, start, goal):
        """Function to compute the shortest path
        Args:  start: (int, int) - start position
               goal: (int, int) - goal position
        """
        # while not finished
        # fetch coordinates u from priority queue

        # if g value at u > rhs value at u:
        # node is over consistent
        # else:
        # node is under consistent

        while self.U.topKey() < self.calculate_key(start, goal) or self.rhs[start] != self.g[start]:
            u = self.U.pop()
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.neighbors8(u):
                    self.update_vertex(s, start, goal)
            else:
                self.g[u] = np.inf
                for s in self.neighbors8(u):
                    self.update_vertex(s, start, goal)
                self.update_vertex(u, start, goal)

            if self.U.empty():
                return

    def reconstruct_path(self, start, goal):
        """Function to reconstruct the path
        Args:  start: (int, int) - start position
               goal: (int, int) - goal position
        Returns:  Path - the path
        """

        path = Path()

        pose = Pose()
        pose.position.x = start[0]+0.5
        pose.position.y = start[1]+0.5
        path.poses.append(pose)

        u = start
        while u != goal:
            for s in self.neighbors8(u):
                if self.g[s] < self.g[u]:
                    u = s
            pose = Pose()
            pose.position.x = u[0]+0.5
            pose.position.y = u[1]+0.5
            path.poses.append(pose)

        return path

    def neighbors8(self, coord):
        """Returns coordinates of passable neighbors of the given cell in 8-neighborhood
        Args:  coord : (int, int) - map coordinate
        Returns:  list (int,int) - list of neighbor coordinates
        """
        neighbours = []

        raw8 = [(coord[0] + 1, coord[1]), (coord[0], coord[1] + 1), (coord[0] - 1, coord[1]), (coord[0], coord[1] - 1),
                (coord[0] + 1, coord[1] + 1), (coord[0] - 1, coord[1] - 1), (coord[0] + 1, coord[1] - 1),
                (coord[0] - 1, coord[1] + 1)]

        for r in raw8:
            if 0 <= r[0] < len(self.gridmap) and 0 <= r[1] < len(self.gridmap[1]) and self.gridmap[r] == 0:
                neighbours.append(r)

        return neighbours

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
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

    def world_to_map(self, p, grid_origin, grid_resolution):
        return ((p - grid_origin) / grid_resolution).astype(int)

    def poseToCell(self, pose: Pose, gridMap: OccupancyGrid) -> tuple:
        return (pose.position.x - gridMap.origin.position.x) / gridMap.resolution, (pose.position.y - gridMap.origin.position.y) / gridMap.resolution

    def cellToPose(self, cell: tuple, gridMap: OccupancyGrid) -> Pose:
        pose = Pose()
        pose.position.x = cell[0] * gridMap.resolution + gridMap.origin.position.x
        pose.position.y = cell[1] * gridMap.resolution + gridMap.origin.position.y
        return pose

    def distanceOfCellsEuclidean(self, cell1: tuple, cell2: tuple) -> float:
        return np.sqrt((cell1[0]-cell2[0])**2 + (cell1[1]-cell2[1])**2)

    def distanceOfPosesAStar(self, pose1: Pose, pose2: Pose, gridMapP: OccupancyGrid) -> float:
        path = self.plan_path(gridMapP, pose1, pose2)
        if path is None:
            print("pose unreachable")
            return 1000
        return len(path.poses)      # uses non-simplified path so we only need to count number of cells

    def cellsSeeEachOther(self, cell1: tuple, cell2: tuple, gridMap: OccupancyGrid) -> bool:
        n = int(self.distanceOfCellsEuclidean(cell1, cell2))
        line = zip(np.linspace(cell1[0], cell2[0], n), np.linspace(cell1[1], cell2[1], n))
        for point in line:
            if gridMap.data[int(point[1])*gridMap.width + int(point[0])] == 1:
                return False
        return True
