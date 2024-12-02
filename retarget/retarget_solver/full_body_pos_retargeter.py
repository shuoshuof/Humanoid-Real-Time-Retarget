# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 14:13
@Auth ： shuoshuof
@File ：hand_retargeter.py
@Project ：Humanoid-Real-Time-Retarget
"""
import torch

from retarget.retarget_solver.base_retargeter import BaseHumanoidRetargeter
from retarget.spatial_transform.transform3d import *
from retarget.robot_config.Hu_v5 import Hu_DOF_AXIS

from robot_kinematics_model import RobotZeroPose, cal_local_rotation


class VtrdynFullBodyPosRetargeter(BaseHumanoidRetargeter):
    def __init__(
            self, mocap_zero_pose: RobotZeroPose,
            target_zero_pose: RobotZeroPose,
            precise_gripper=False
    ):
        super().__init__(mocap_zero_pose, target_zero_pose)
        self.precise_gripper = precise_gripper
    def retarget(
            self,
            body_global_translation,
            left_hand_global_translation,
            right_hand_global_translation
    ):
        robot_local_rotation = quat_identity_like(self.target_zero_pose.local_rotation)
        body_global_rotation = quat_identity_like(self.source_zero_pose.global_rotation)

        robot_local_rotation,body_global_rotation = self._retarget_arm_from_global_translation(
            robot_local_rotation,
            body_global_rotation,
            body_global_translation,
        )

        robot_local_rotation,body_global_rotation = self._retarget_wrist_from_global_translation(
            robot_local_rotation,
            body_global_rotation,
            left_hand_global_translation,
            right_hand_global_translation
        )

        dof_pos = quat_to_dof_pos(robot_local_rotation[1:], Hu_DOF_AXIS)

        dof_pos = self._retarget_gripper(
            dof_pos,
            body_global_rotation,
            left_hand_global_translation,
            right_hand_global_translation
        )

        self._motion_local_rotation.append(robot_local_rotation)
        self._motion_dof_pos.append(dof_pos)

        return robot_local_rotation, dof_pos, body_global_rotation

    def _retarget_arm_from_global_translation(
            self,
            robot_local_rotation,
            body_global_rotation,
            body_global_translation

    ):
        joint10_global_rotation = cal_joint_quat(
            self.source_zero_pose.local_translation[[11, 36, 34]].unsqueeze(0), # full indices
            (body_global_translation[[17, 13, 11]] - body_global_translation[[10]]).unsqueeze(0)
        )

        left_shoulder_parent_quat = joint10_global_rotation

        left_shoulder_pitch, left_shoulder_roll = cal_shoulderPR(
            body_global_translation[19] - body_global_translation[18],
            # self.source_zero_pose.local_translation[19],
            self.source_zero_pose.local_translation[13],  # full
            left_shoulder_parent_quat
        )

        left_elbow_parent_quat = quat_mul_three(left_shoulder_parent_quat, left_shoulder_pitch, left_shoulder_roll)

        left_shoulder_yaw, left_elbow_pitch = cal_elbowP_and_shoulderY(
            body_global_translation[20] - body_global_translation[19],
            self.source_zero_pose.local_translation[14],  # full
            left_elbow_parent_quat
        )

        robot_local_rotation[12] = left_shoulder_pitch
        robot_local_rotation[13] = left_shoulder_roll
        robot_local_rotation[14] = left_shoulder_yaw
        robot_local_rotation[15] = left_elbow_pitch

        right_shoulder_parent_quat = joint10_global_rotation

        right_shoulder_pitch, right_shoulder_roll = cal_shoulderPR(
            body_global_translation[15] - body_global_translation[14],
            self.source_zero_pose.local_translation[38],
            right_shoulder_parent_quat
        )

        right_elbow_parent_quat = quat_mul_three(right_shoulder_parent_quat, right_shoulder_pitch, right_shoulder_roll)

        right_shoulder_yaw, right_elbow_pitch = cal_elbowP_and_shoulderY(
            body_global_translation[16] - body_global_translation[15],
            self.source_zero_pose.local_translation[39],
            right_elbow_parent_quat
        )

        robot_local_rotation[21] = right_shoulder_pitch
        robot_local_rotation[22] = right_shoulder_roll
        robot_local_rotation[23] = right_shoulder_yaw
        robot_local_rotation[24] = right_elbow_pitch

        body_global_rotation[10] = joint10_global_rotation

        return robot_local_rotation,body_global_rotation

    def _retarget_wrist_from_global_translation(
            self,
            robot_local_rotation,
            body_global_rotation,
            left_hand_global_translation,
            right_hand_global_translation
    ):
        # cal left wrist
        left_wrist_parent_quat = quat_mul_norm(
            body_global_rotation[10],
            quat_mul_four(
                robot_local_rotation[12],
                robot_local_rotation[13],
                robot_local_rotation[14],
                robot_local_rotation[15],
            )
        )

        left_wrist_global_quat = cal_joint_quat(
            self.source_zero_pose.local_translation[[16, 20, 24, 28, 32]].unsqueeze(0),
            left_hand_global_translation[[2, 6, 10, 14, 17]]-left_hand_global_translation[[0]].unsqueeze(0)
        )

        left_wrist_local_quat = quat_mul_norm(quat_inverse(left_wrist_parent_quat), left_wrist_global_quat)

        left_elbow_roll, left_wrist_pitch, left_wrist_yaw = quat_in_xyz_axis(left_wrist_local_quat, seq='XYZ')
        robot_local_rotation[16] = left_elbow_roll
        robot_local_rotation[17] = left_wrist_pitch
        robot_local_rotation[18] = left_wrist_yaw

        # cal right wrist
        right_wrist_parent_quat = quat_mul_norm(
            body_global_rotation[10],
            quat_mul_four(
                robot_local_rotation[21],
                robot_local_rotation[22],
                robot_local_rotation[23],
                robot_local_rotation[24],
            )
        )

        right_wrist_global_quat = cal_joint_quat(
            self.source_zero_pose.local_translation[[41, 45, 49, 53, 56]].unsqueeze(0),
            right_hand_global_translation[[2, 6, 10, 14, 17]]-right_hand_global_translation[[0]].unsqueeze(0)
        )

        right_wrist_local_quat = quat_mul_norm(quat_inverse(right_wrist_parent_quat), right_wrist_global_quat)
        right_elbow_roll, right_wrist_pitch, right_wrist_yaw = quat_in_xyz_axis(right_wrist_local_quat, seq='XYZ')
        robot_local_rotation[25] = right_elbow_roll
        robot_local_rotation[26] = right_wrist_pitch
        robot_local_rotation[27] = right_wrist_yaw

        body_global_rotation[14] = left_wrist_global_quat
        body_global_rotation[39] = right_wrist_global_quat

        return robot_local_rotation, body_global_rotation

    def _retarget_gripper(
            self,
            dof_pos,
            body_global_rotation,
            left_hand_global_translation,
            right_hand_global_translation,
    ):
        orig_hand_x_dist_to_wrist = self.source_zero_pose.global_translation[[18,22,26,30,33],0]-self.source_zero_pose.global_translation[14,0]
        orig_hand_avg_x_dist = orig_hand_x_dist_to_wrist.mean()

        left_wrist_global_quat = body_global_rotation[14]
        # transform to left wrist frame
        left_hand_global_translation = quat_rotate(quat_inverse(left_wrist_global_quat), left_hand_global_translation)
        left_hand_x_dist_to_wrist = (left_hand_global_translation-left_hand_global_translation[0])[[4,8,12,16,19],0]

        right_wrist_global_quat = body_global_rotation[39]
        # transform to right wrist frame
        right_hand_global_translation = quat_rotate(quat_inverse(right_wrist_global_quat), right_hand_global_translation)
        right_hand_x_dist_to_wrist = (right_hand_global_translation-right_hand_global_translation[0])[[4,8,12,16,19],0]
        left_avg_x_dist = left_hand_x_dist_to_wrist.mean()
        right_avg_x_dist = right_hand_x_dist_to_wrist.mean()

        if self.precise_gripper:
            left_stretch = torch.clip(left_avg_x_dist/orig_hand_avg_x_dist - 0.5,0,0.5)/0.5
            dof_pos[19 - 1] = left_stretch*0.044
            dof_pos[20 - 1] = left_stretch*-0.044

            right_stretch = torch.clip(right_avg_x_dist/orig_hand_avg_x_dist - 0.5,0,0.5)/0.5
            dof_pos[28-1] = right_stretch*0.044
            dof_pos[29-1] = right_stretch*-0.044
        else:
            left_close = left_avg_x_dist/orig_hand_avg_x_dist < 0.7
            right_close = right_avg_x_dist/orig_hand_avg_x_dist < 0.7

            dof_pos[19 - 1] = 0 if left_close else 0.044
            dof_pos[20 - 1] = 0 if left_close else -0.044

            dof_pos[28-1] = 0 if right_close else 0.044
            dof_pos[29-1] = 0 if right_close else -0.044
            # print(left_close,right_close)
        return dof_pos


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
    import pickle
    import pandas as pd
    import time
    from robot_kinematics_model import RobotZeroPose
    from vedo_visualizer.common import vis_zero_pose, vis_robots

    from retarget.torch_ext import to_torch
    from retarget.utils.parse_mocap import get_vtrdyn_full_translation, get_vtrdyn_full_rotation
    from retarget.utils.parse_mocap import vtrdyn_full_zero_pose_transform

    with open('asset/zero_pose/vtrdyn_full_zero_pose.pkl', 'rb') as f:
        vtrdyn_full_zero_pose = pickle.load(f)

    vtrdyn_full_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_full_zero_pose)
    # vis_zero_pose([vtrdyn_full_zero_pose])

    df = pd.read_csv('test_motion/mocap_raw/motionData_20241125200200.csv')
    motion_global_translation = to_torch(get_vtrdyn_full_translation(df))
    motion_global_rotation = to_torch(get_vtrdyn_full_rotation(df))
    motion_global_rotation = vtrdyn_full_zero_pose_transform(global_rotation=motion_global_rotation)

    colors = np.zeros_like(motion_global_translation) + np.array([1, 0, 0])
    # colors[...,14:14+20,:] = np.array([0, 1, 0])
    colors[...,[16, 20, 24, 28, 31],:] = np.array([0, 1, 0])
    # colors[...,39:39+20,:] = np.array([0, 0, 1])
    colors[...,[41, 45, 49, 53, 56],:] = np.array([0, 0, 1])
    mocap_data = [{'body_pos': body_pos, 'body_quat': body_quat,'color':color} for body_pos, body_quat,color in
                  zip(motion_global_translation, motion_global_rotation,colors)]
    # vis_robots([mocap_data],[vtrdyn_full_zero_pose])

    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')
    # vis_zero_pose([hu_zero_pose])

    hu_retarget = VtrdynFullBodyPosRetargeter(vtrdyn_full_zero_pose, hu_zero_pose)

    for i in range(500):
        start = time.time()
        global_translation = motion_global_translation[i]
        body_global_translation = global_translation[
            [0,  4, 5, 6,  1, 2, 3,  7, 8, 9,  10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]]
        left_hand_global_translation = global_translation[14:14 + 20]
        right_hand_global_translation = global_translation[39:39 + 20]

        global_rotation = motion_global_rotation[i]
        body_global_rotation = global_rotation[
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]]
        # left_hand_global_rotation = global_rotation[14:14 + 20]
        # right_hand_global_rotation = global_rotation[39:39 + 20]

        _,_, cal_body_global_rotation = hu_retarget.retarget(
            body_global_translation,
            # left_hand_global_rotation,
            left_hand_global_translation,
            # right_hand_global_rotation,
            right_hand_global_translation
        )
        print((cal_body_global_rotation[...,14,:4]-body_global_rotation[...,20,:4]).max())
        end = time.time()
        print(f'Time cost {end - start:.5f} s')

    robot_data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
                  zip(hu_retarget.motion_global_translation, hu_retarget.motion_global_rotation)]

    # vis_robots([robot_data, mocap_data], [hu_zero_pose, vtrdyn_full_zero_pose])
    vis_robots([robot_data], [hu_zero_pose])
