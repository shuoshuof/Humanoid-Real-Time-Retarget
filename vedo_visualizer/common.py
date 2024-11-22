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

import torch

from robot_kinematics_model.base_robot import RobotZeroPose
from vedo_visualizer import SkeletonRobotVisualizer,BaseRobot,BaseSkeletonRobot

def vis_mocap_robot(motions:List):
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    robot = BaseSkeletonRobot.from_zero_pose(vtrdyn_zero_pose)
    robots = [copy.deepcopy(robot) for _ in range(len(motions))]

    vis = SkeletonRobotVisualizer(len(motions),robots,motions)
    vis.show()

def vis_zero_pose(zero_poses:List[RobotZeroPose]):
    robots = [BaseSkeletonRobot.from_zero_pose(zero_pose) for zero_pose in zero_poses]
    data = [[{'body_pos':zero_pose.global_translation,'body_quat':zero_pose.global_rotation}] for zero_pose in zero_poses]
    vis = SkeletonRobotVisualizer(len(zero_poses),robots,data)
    vis.show()

if __name__ == "__main__":
    from poselib.poselib.core.rotation3d import *
    from retarget.spatial_transform.transform3d import *
    from retarget.utils import get_mocap_rotation,get_mocap_translation
    from retarget.torch_ext import *
    import pandas as pd
    import numpy as np

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')

    motion_global_translation = get_mocap_translation(df)
    motion_global_rotation = get_mocap_rotation(df)
    motion_global_translation = motion_global_translation-motion_global_translation[0,0]

    rotation = quat_from_angle_axis(torch.tensor(torch.pi/2),torch.tensor([0.,0.,1]))
    motion_global_rotation = to_numpy(quat_mul_norm(to_torch(motion_global_rotation),rotation))


    data = [{'body_pos':body_pos,'body_quat':body_quat} for body_pos, body_quat in zip(motion_global_translation, motion_global_rotation)]

    vis_mocap_robot([data,data])

    # with open('asset/t_pose/vtrdyn_t_pose.pkl', 'rb') as f:
    #     vtrdyn_t_pose = pickle.load(f)
    # zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_t_pose)
    # vis_zero_pose([zero_pose])




