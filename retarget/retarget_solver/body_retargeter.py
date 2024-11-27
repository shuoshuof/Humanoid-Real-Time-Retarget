# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/22 16:23
@Auth ： shuoshuof
@File ：body_retargeter.py
@Project ：Humanoid-Real-Time-Retarget
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import time

import pickle

import torch

from robot_kinematics_model import RobotZeroPose, cal_local_rotation, cal_forward_kinematics

from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch, to_numpy
from retarget.retarget_solver.base_retargeter import BaseHumanoidRetargeter

from retarget.robot_config.Hu_v5 import Hu_DOF_AXIS

from vedo_visualizer.common import vis_robots


class Mocap2HuBodyRetargeter(BaseHumanoidRetargeter):
    def __init__(self, mocap_zero_pose: RobotZeroPose, target_zero_pose: RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)

    def retarget_from_pose(self, source_global_rotation: torch.Tensor):
        source_local_rotation = cal_local_rotation(to_torch(source_global_rotation),
                                                   self.source_zero_pose.parent_indices)

        robot_local_rotation = quat_identity_like(self.target_zero_pose.local_rotation)

        left_shoulder_quat = source_local_rotation[18]
        # left_shoulder_roll,left_shoulder_pitch,left_shoulder_yaw = quat_in_xyz_axis(left_shoulder_quat,'YXZ')
        left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw = quat_in_xyz_axis(left_shoulder_quat, 'YXZ')

        right_shoulder_quat = source_local_rotation[14]
        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw = quat_in_xyz_axis(right_shoulder_quat, 'YXZ')

        left_elbow_quat = source_local_rotation[19]
        # left_elbow_pitch, left_elbow_roll, left_elbow_yaw = quat_in_xyz_axis(left_elbow_quat, 'ZYX')
        left_elbow_yaw,left_elbow_pitch, left_elbow_roll = quat_in_xyz_axis(left_elbow_quat,'ZYX')
        # left_elbow_pitch, left_elbow_yaw, left_elbow_roll = quat_in_xyz_axis(left_elbow_quat,'YZX')
        # left_elbow_yaw, left_elbow_roll,left_elbow_pitch = quat_in_xyz_axis(left_elbow_quat,'ZXY')

        right_elbow_quat = source_local_rotation[15]
        # right_elbow_pitch, right_elbow_roll, right_elbow_yaw = quat_in_xyz_axis(right_elbow_quat, 'ZYX')
        right_elbow_yaw,right_elbow_pitch, right_elbow_roll = quat_in_xyz_axis(right_elbow_quat,'ZYX')
        # right_elbow_pitch,right_elbow_yaw, right_elbow_roll = quat_in_xyz_axis(right_elbow_quat,'YZX')
        # right_elbow_yaw, right_elbow_roll, right_elbow_pitch = quat_in_xyz_axis(right_elbow_quat,'ZXY')

        robot_local_rotation[12] = left_shoulder_pitch
        robot_local_rotation[13] = left_shoulder_roll
        # robot_local_rotation[14] = quat_mul_norm(left_shoulder_yaw,left_elbow_yaw)
        robot_local_rotation[14] = quat_mul_norm(left_elbow_yaw,left_shoulder_yaw)

        robot_local_rotation[21] = right_shoulder_pitch
        robot_local_rotation[22] = right_shoulder_roll
        # robot_local_rotation[23] = quat_mul_norm(right_shoulder_yaw,right_elbow_yaw)
        robot_local_rotation[23] = quat_mul_norm(right_elbow_yaw,right_shoulder_yaw)


        robot_local_rotation[15] = left_elbow_pitch
        robot_local_rotation[16] = left_elbow_roll

        robot_local_rotation[24] = right_elbow_pitch
        robot_local_rotation[25] = right_elbow_roll

        dof_pos = quat_to_dof_pos(robot_local_rotation[1:], Hu_DOF_AXIS)

        self._motion_local_rotation.append(robot_local_rotation)
        self._motion_dof_pos.append(dof_pos)

        return robot_local_rotation, dof_pos

    def retarget_test(self, source_global_rotation):
        source_local_rotation = cal_local_rotation(to_torch(source_global_rotation),
                                                   self.source_zero_pose.parent_indices)

        robot_local_rotation = quat_identity_like(self.target_zero_pose.local_rotation)

        robot_local_rotation[13] = source_local_rotation[18]
        robot_local_rotation[15] = source_local_rotation[19]
        robot_local_rotation[22] = source_local_rotation[14]
        robot_local_rotation[24] = source_local_rotation[15]

        dof_pos = quat_to_dof_pos(robot_local_rotation[1:], Hu_DOF_AXIS)

        self._motion_local_rotation.append(robot_local_rotation)
        self._motion_dof_pos.append(dof_pos)

        return robot_local_rotation, dof_pos


if __name__ == '__main__':
    from retarget.utils.parse_mocap import get_vtrdyn_translation,get_vtrdyn_rotation
    from retarget.utils.parse_mocap import vtrdyn_zero_pose_transform

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = to_torch(get_vtrdyn_translation(df))
    motion_global_rotation = to_torch(get_vtrdyn_rotation(df))
    motion_global_rotation = vtrdyn_zero_pose_transform(global_rotation=motion_global_rotation)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)
    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)

    data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
            zip(motion_global_translation, motion_global_rotation)]

    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    hu_retarget = Mocap2HuBodyRetargeter(vtrdyn_zero_pose, hu_zero_pose)

    for i in range(300):
        start = time.time()
        robot_local_rotation, dof_pos = hu_retarget.retarget_from_pose(motion_global_rotation[i])
        # robot_local_rotation, dof_pos = hu_retarget.retarget_test(motion_global_rotation[i])
        end = time.time()
        print(f'Time cost {end - start:.5f} s')


    data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
            zip(to_numpy(hu_retarget.motion_global_translation), to_numpy(hu_retarget.motion_global_rotation))]

    vis_robots([data], hu_zero_pose)

    # motion = SkeletonMotion.from_skeleton_state(state,fps=30)
    # plot_skeleton_H([motion])
