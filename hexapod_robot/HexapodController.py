#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from messages import *

DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10


class HexapodController:
    def __init__(self):
        pass

    def goto(self, goal, odometry, collision):
        """Method to steer the robot towards the goal position given its current 
           odometry and collision status
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = Twist()

        if collision:
            return None

        if goal is not None and odometry is not None:
            diff = goal.position - odometry.pose.position
            target_to_goal = diff.norm()
            is_in_goal = target_to_goal < DELTA_DISTANCE
            if is_in_goal:
                return None

            targ_heading = np.arctan2(diff.y, diff.x)
            cur_heading = odometry.pose.orientation.to_Euler()[0]  # quaternion -> euler angle

            diff_h = targ_heading - cur_heading
            diff_h = (diff_h + math.pi) % (2 * math.pi) - math.pi

            lin_mult = 0 if abs(diff_h) > math.pi / 6 else 500
            turn_mult = 500 if abs(diff_h) > math.pi / 6 else C_TURNING_SPEED

            cmd_msg.linear.x = target_to_goal * lin_mult
            cmd_msg.angular.z = diff_h * turn_mult

        return cmd_msg

    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding 
           contact with the obstacles given its current odometry, collision 
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = self.goto(goal, odometry, collision)

        if laser_scan is not None and cmd_msg is not None:
            scan_left = np.min(laser_scan.distances[:len(
                laser_scan.distances) // 2])  # distance to the closest obstacle to the left of the robot
            scan_right = np.min(laser_scan.distances[
                                len(laser_scan.distances) // 2:])  # distance to the closest obstacle to the right of the robot
            repulsive_force = 1 / scan_left - 1 / scan_right

            cmd_msg.angular.z += repulsive_force * C_AVOID_SPEED

        return cmd_msg
