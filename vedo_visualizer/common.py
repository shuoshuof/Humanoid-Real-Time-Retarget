# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/21 17:28
@Auth ： shuoshuof
@File ：common.py
@Project ：Humanoid-Real-Time-Retarget
"""
import copy
from typing import List
import pickle

from robot_kinematics_model.base_robot import RobotZeroPose
from vedo_visualizer import SkeletonRobotVisualizer,BaseSkeletonRobot,MocapRobot

def vis_mocap_robot(motions:List):
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    robot = MocapRobot.from_zero_pose(vtrdyn_zero_pose)
    robots = [copy.deepcopy(robot) for _ in range(len(motions))]

    vis = SkeletonRobotVisualizer(len(motions),robots,motions)
    vis.show()


if __name__ == "__main__":
    with open('data/2024-11-21_18:03:02.pkl','rb') as f:
        motion = pickle.load(f)
    body_pos = motion['body_pos']
    body_pos-=body_pos[0,0]
    vis_mocap_robot([body_pos])






