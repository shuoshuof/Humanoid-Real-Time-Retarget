# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 12:00
@Auth ： shuoshuof
@File ：parse_mocap.py
@Project ：Humanoid-Real-Time-Retarget
"""
import pandas as pd
import numpy as np
import pickle

from poselib.poselib.core.rotation3d import *

from retarget.robot_config import VTRDYN
from retarget.robot_config import VTRDYN_FULL
from poselib.poselib.core.rotation3d import *
from robot_kinematics_model.kinematics import cal_local_rotation
from retarget.spatial_transform.transform3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from retarget.torch_ext import *


from vedo_visualizer import vis_robot, vis_zero_pose
from robot_kinematics_model.base_robot import RobotZeroPose

def get_vtrdyn_translation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN.VTRDYN_JOINT_NAMES), 3))
    for joint_idx, joint_name in enumerate(VTRDYN.VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} position X(m)']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} position Y(m)']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} position Z(m)']
    return motion_global_translation

def get_vtrdyn_rotation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN.VTRDYN_JOINT_NAMES), 4))
    for joint_idx, joint_name in enumerate(VTRDYN.VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} quaternion X']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} quaternion Y']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} quaternion Z']
        motion_global_translation[:, joint_idx, 3] = data[f'{joint_name} quaternion W']
    return motion_global_translation

def get_vtrdyn_full_translation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_FULL.VTRDYN_JOINT_NAMES), 3))
    for joint_idx, joint_name in enumerate(VTRDYN_FULL.VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} position X(m)']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} position Y(m)']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} position Z(m)']
    return motion_global_translation

def get_vtrdyn_full_rotation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_FULL.VTRDYN_JOINT_NAMES), 4))
    for joint_idx, joint_name in enumerate(VTRDYN_FULL.VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} quaternion X']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} quaternion Y']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} quaternion Z']
        motion_global_translation[:, joint_idx, 3] = data[f'{joint_name} quaternion W']
    return motion_global_translation


with open('asset/t_pose/vtrdyn_full_t_pose.pkl', 'rb') as f:
    vtrdyn_full_t_pose: SkeletonState = pickle.load(f)

vtrdyn_full_zero_pose = RobotZeroPose.from_skeleton_state(
    skeleton_state=vtrdyn_full_t_pose)

zero_pose_local_rotation = vtrdyn_full_zero_pose.local_rotation.clone()

zero_pose_local_rotation[12] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[13] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([0., 0, 1.]))
zero_pose_local_rotation[37] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[38] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1.]))

vtrdyn_full_t2zero_pose_transform_quat = vtrdyn_full_zero_pose.rebuild_pose_by_local_rotation(zero_pose_local_rotation)


def vtrdyn_full_zero_pose_transform(global_rotation):
    rotation = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1]))
    transformed_global_rotation = global_rotation

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation, rotation)

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation,quat_inverse(vtrdyn_full_t2zero_pose_transform_quat))

    return transformed_global_rotation

with open('asset/t_pose/vtrdyn_t_pose.pkl', 'rb') as f:
    vtrdyn_t_pose: SkeletonState = pickle.load(f)

vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(
    skeleton_state=vtrdyn_t_pose)

zero_pose_local_rotation = vtrdyn_zero_pose.local_rotation.clone()
zero_pose_local_rotation[18] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[19] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([0., 0, 1.]))
zero_pose_local_rotation[14] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[15] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1.]))

# t pose to zero pose transform
vtrdyn_t2zero_pose_transform_quat = vtrdyn_zero_pose.rebuild_pose_by_local_rotation(zero_pose_local_rotation)

def vtrdyn_zero_pose_transform(global_rotation):
    rotation = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1]))
    transformed_global_rotation = global_rotation

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation, rotation)

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation,quat_inverse(vtrdyn_t2zero_pose_transform_quat))

    return transformed_global_rotation

if __name__ == '__main__':

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')

    motion_global_translation = get_vtrdyn_translation(df)
    motion_global_rotation = get_vtrdyn_rotation(df)
    motion_global_translation = motion_global_translation - motion_global_translation[0, 0]

    with open('asset/t_pose/vtrdyn_full_t_pose.pkl', 'rb') as f:
        vtrdyn_t_pose: SkeletonState = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(
        skeleton_state=vtrdyn_t_pose)

    zero_pose_local_rotation = vtrdyn_zero_pose.local_rotation.clone()

    zero_pose_local_rotation[12] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([1., 0., 0.]))
    zero_pose_local_rotation[13] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([0., 0, 1.]))
    zero_pose_local_rotation[37] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([1., 0., 0.]))
    zero_pose_local_rotation[38] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1.]))

    vtrdyn_full_t2zero_pose_transform_quat = vtrdyn_zero_pose.rebuild_pose_by_local_rotation(zero_pose_local_rotation)
    vis_zero_pose([vtrdyn_zero_pose])

