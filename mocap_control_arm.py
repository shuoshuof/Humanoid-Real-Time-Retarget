# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/11 19:56
@Auth ： shuoshuof
@File ：mocap_control_arm.py
@Project ：Humanoid-Real-Time-Retarget
"""
import numpy as np
from vedo_visualizer import vis_robots
import pickle
from robot_kinematics_model import RobotZeroPose

from retarget.torch_ext import *

from retarget.spatial_transform.transform3d import *

def cal_root_rot(body_global_translation,vtrdyn_zero_pose):
    joint10_global_rotation = cal_joint_quat(
        vtrdyn_zero_pose.local_translation[[10, 13, 17]].unsqueeze(0),  # full indices
        (body_global_translation[[10, 13, 17]] - body_global_translation[[10]]).unsqueeze(0)
    )
    return joint10_global_rotation

if __name__ == '__main__':
    with open('data/test_pos(1).pkl','rb') as f:
        motion_body_pos = pickle.load(f)
    motion_body_pos = np.array(motion_body_pos)[:,[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],:]
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)

    color = np.zeros_like(motion_body_pos) + np.array([1, 1, 1])
    color[:,10] = np.array([1,0,0])
    color[:, 13] = np.array([0, 1, 0])
    color[:, 17] = np.array([0, 0, 1])

    motion_body_pos = to_torch(motion_body_pos)
    base_quat = cal_root_rot(motion_body_pos[0],vtrdyn_zero_pose)
    motion_body_pos = quat_rotate(quat_inverse(base_quat),(motion_body_pos-motion_body_pos[0,0]))
    motion_body_pos =  to_numpy(motion_body_pos)


    data = [{'body_pos': body_pos, "color":c} for body_pos,c in zip(motion_body_pos,color)]

    vis_robots([data], [vtrdyn_zero_pose])
