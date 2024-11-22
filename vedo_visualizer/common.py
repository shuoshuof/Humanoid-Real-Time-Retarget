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
from vedo_visualizer import SkeletonRobotVisualizer,BaseRobot,BaseSkeletonRobot

def vis_mocap_robot(motions:List):
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    robot = BaseSkeletonRobot.from_zero_pose(vtrdyn_zero_pose)
    robots = [copy.deepcopy(robot) for _ in range(len(motions))]

    vis = SkeletonRobotVisualizer(len(motions),robots,motions)
    vis.show()



if __name__ == "__main__":
    from retarget.utils import get_mocap_rotation,get_mocap_translation
    import pandas as pd
    import numpy as np
    # with open('data/2024-11-21_19:56:13.pkl','rb') as f:
    #     motion = pickle.load(f)
    # body_pos = motion['body_pos']
    # body_pos-=body_pos[0,0]
    # vis_mocap_robot([body_pos,body_pos])

    df = pd.read_csv('test_motion/mocap_raw/test1.csv')
    # get_motion_quaternions(df)
    motion_global_translation = get_mocap_translation(df)
    motion_global_rotation = get_mocap_rotation(df)
    motion_global_translation = motion_global_translation-motion_global_translation[0,0]
    # motion_data = np.concatenate([motion_global_translation, motion_global_rotation], axis=1)
    data = [{'body_pos':body_pos,'body_quat':body_quat} for body_pos, body_quat in zip(motion_global_translation, motion_global_rotation)]

    vis_mocap_robot([data])




