# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/20 09:38
@Auth ： shuoshuof
@File ：retarget.py
@Project ：Humanoid-Real-Time-Retarget
"""

from abc import ABC
import pandas as pd
import time

import pickle

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.poselib.visualization.common import plot_skeleton_H

from robot_kinematics_model import RobotZeroPose

from retarget.retarget_solver.base_retargeter import BaseHumanoidRetargeter
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch

from retarget.robot_config.Hu_v5 import Hu_DOF_AXIS


class HuUpperBodyFromMocapRetarget(BaseHumanoidRetargeter):
    def __init__(self, mocap_zero_pose: RobotZeroPose, target_zero_pose: RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)
        self._motion_local_rotation = []
        self._motion_dof_pos = []
        # self.filter = WeightedFilter(0.5)

    def _cal_shoulderPR(self, v1, v0, parent_global_rotation):
        return cal_shoulderPR(v1, v0, parent_global_rotation)

    def _cal_elbowP_and_shoulderY(self, v1, v0, parent_global_rotation):
        return cal_elbowP_and_shoulderY(v1, v0, parent_global_rotation)

    def retarget_from_global_translation(self, source_global_translation: torch.Tensor):
        source_global_translation = coord_transform(source_global_translation, dir=torch.Tensor([-1, -1, 1]))

        # source_global_translation = self.rescale_motion_to_standard_size(source_global_translation, self.source_zero_pose)
        # mocap_motion = self._rebuild_with_vtrdyn_zero_pose(motion_global_translation)

        robot_local_rotation = quat_identity_like(self.target_zero_pose.local_rotation)

        # mocap cheat global rotation
        joint10_global_rotation = cal_joint_quat(
            self.source_zero_pose.local_translation[[17, 13, 11]].unsqueeze(0),
            (source_global_translation[[17, 13, 11]] - source_global_translation[[10]]).unsqueeze(0)
        )

        left_shoulder_pitch, left_shoulder_roll = self._cal_shoulderPR(
            source_global_translation[19] - source_global_translation[18],
            self.source_zero_pose.local_translation[19],
            joint10_global_rotation
        )

        right_shoulder_pitch, right_shoulder_roll = self._cal_shoulderPR(
            source_global_translation[15] - source_global_translation[14],
            self.source_zero_pose.local_translation[15],
            joint10_global_rotation
        )

        robot_local_rotation[12] = left_shoulder_pitch
        robot_local_rotation[13] = left_shoulder_roll

        robot_local_rotation[21] = right_shoulder_pitch
        robot_local_rotation[22] = right_shoulder_roll

        left_elbow_parent_quat = quat_mul_three(joint10_global_rotation, left_shoulder_pitch, left_shoulder_roll)

        left_shoulder_yaw, left_elbow_pitch = self._cal_elbowP_and_shoulderY(
            source_global_translation[20] - source_global_translation[19],
            self.source_zero_pose.local_translation[20],
            left_elbow_parent_quat
        )

        right_elbow_parent_quat = quat_mul_three(joint10_global_rotation, right_shoulder_pitch, right_shoulder_roll)

        right_shoulder_yaw, right_elbow_pitch = self._cal_elbowP_and_shoulderY(
            source_global_translation[16] - source_global_translation[15],
            self.source_zero_pose.local_translation[16],
            right_elbow_parent_quat
        )

        robot_local_rotation[14] = left_shoulder_yaw
        robot_local_rotation[15] = left_elbow_pitch
        robot_local_rotation[23] = right_shoulder_yaw
        robot_local_rotation[24] = right_elbow_pitch

        dof_pos = quat_to_dof_pos(robot_local_rotation[1:], Hu_DOF_AXIS)
        # dof_pos = self.filter.filter(dof_pos)

        self._motion_local_rotation.append(robot_local_rotation)
        self._motion_dof_pos.append(dof_pos)

        return robot_local_rotation, dof_pos


@torch.jit.script
def cal_elbowP_and_shoulderY(v1, v0, parent_global_rotation):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    axis = torch.eye(3, dtype=torch.float32)
    parent_quat_inv = quat_inverse(parent_global_rotation)

    v1 = quat_rotate(parent_quat_inv, v1).squeeze(0)

    # v1 proj in xoy plane
    v1_proj = proj_in_plane(v1, axis[2])
    v0_proj = proj_in_plane(v0, axis[2])

    # shoulder yaw
    theta1 = radians_between_vecs(axis[0], v1_proj, n=axis[2])
    theta0 = radians_between_vecs(axis[0], v0_proj, n=axis[2])

    shoulder_yaw_quat = quat_from_angle_axis(theta1 - theta0, axis[2])

    # elbow pitch
    phi1 = radians_between_vecs(v1_proj, v1, n=torch.cross(axis[2], v1_proj))
    phi0 = radians_between_vecs(v0_proj, v0, n=torch.cross(axis[2], v0_proj))

    elbow_pitch_quat = quat_from_angle_axis(phi1 - phi0, axis[1])
    return shoulder_yaw_quat, elbow_pitch_quat
@torch.jit.script
def cal_shoulderPR(v1, v0, parent_global_rotation):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    r"""
    calculate shoulder spherical joint rotation
    the order is pitch -> roll. the raw need to use the elbow to cal
    :param v1:
    :param v0:
    :param parent_global_rotation:
    :param plane:
    :return:
    """
    axis = torch.eye(3, dtype=torch.float32)

    parent_quat_inv = quat_inverse(parent_global_rotation)

    v1 = quat_rotate(parent_quat_inv, v1).squeeze(0)

    # v1 proj in xoz plane
    v1_proj = proj_in_plane(v1, axis[1])

    v0_proj = proj_in_plane(v0, axis[1])

    theta1 = radians_between_vecs(axis[0], v1_proj, n=axis[1])
    theta0 = radians_between_vecs(axis[0], v0_proj, n=axis[1])

    pitch_joint_quat = quat_from_angle_axis(theta1 - theta0, axis[1])

    phi1 = radians_between_vecs(v1_proj, v1, n=torch.cross(v1_proj, axis[1]))
    phi0 = radians_between_vecs(v0_proj, v0, n=torch.cross(v0_proj, axis[1]))

    roll_joint_quat = quat_from_angle_axis(phi1 - phi0, axis[0])
    return pitch_joint_quat, roll_joint_quat


if __name__ == '__main__':
    from retarget.utils.parse_mocap import get_vtrdyn_translation
    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = get_vtrdyn_translation(df)
    motion_global_translation = to_torch(motion_global_translation)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    hu_retarget = HuUpperBodyFromMocapRetarget(vtrdyn_zero_pose, hu_zero_pose)

    for i in range(200):
        start = time.time()
        hu_retarget.retarget_from_global_translation(motion_global_translation[i])
        end = time.time()
        print(f'Time cost {end - start:.5f} s')

    state = SkeletonState.from_rotation_and_root_translation(
        hu_zero_pose.skeleton_tree,
        hu_retarget.motion_local_rotation,
        torch.zeros((hu_retarget.motion_length, 3)),
        is_local=True
    )

    motion = SkeletonMotion.from_skeleton_state(state, fps=30)
    plot_skeleton_H([motion])
