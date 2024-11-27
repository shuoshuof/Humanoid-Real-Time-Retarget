# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 14:13
@Auth ： shuoshuof
@File ：hand_retargeter.py
@Project ：Humanoid-Real-Time-Retarget
"""
from retarget.retarget_solver.base_retargeter import BaseHumanoidRetargeter
from retarget.spatial_transform.transform3d import *
from retarget.robot_config.Hu_v5 import Hu_DOF_AXIS

from robot_kinematics_model import RobotZeroPose, cal_local_rotation


class FullBodyRetargeter(BaseHumanoidRetargeter):
    def __init__(self, mocap_zero_pose: RobotZeroPose, target_zero_pose: RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)

    def retarget(
            self,
            body_global_rotation,
            body_global_translation,
            left_hand_global_rotation,
            left_hand_global_translation,
            right_hand_global_rotation,
            right_hand_global_translation
    ):
        robot_local_rotation = quat_identity_like(self.target_zero_pose.local_rotation)

        robot_local_rotation = self._retarget_arm_from_global_translation(
            body_global_rotation, body_global_translation,robot_local_rotation)

        robot_local_rotation = self._retarget_wrist_from_global_rotation(robot_local_rotation, body_global_rotation)

        dof_pos = quat_to_dof_pos(robot_local_rotation[1:], Hu_DOF_AXIS)

        self._motion_local_rotation.append(robot_local_rotation)
        self._motion_dof_pos.append(dof_pos)

        return robot_local_rotation, dof_pos

    def _retarget_arm_from_global_translation(self, body_global_rotation, body_global_translation,
                                              robot_local_rotation):
        left_shoulder_parent_quat = body_global_rotation[17]

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

        right_shoulder_parent_quat = body_global_rotation[13]

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

        return robot_local_rotation

    def _retarget_arm_from_global_rotation(self, source_global_rotation, robot_local_rotation):
        pass

    def _retarget_wrist_from_global_rotation(
            self,
            robot_local_rotation,
            body_global_rotation,
    ):
        # cal left wrist
        left_wrist_parent_quat = quat_mul_norm(
            body_global_rotation[17],
            quat_mul_four(
                robot_local_rotation[12],
                robot_local_rotation[13],
                robot_local_rotation[14],
                robot_local_rotation[15],
            )
        )
        left_wrist_local_quat = quat_mul_norm(quat_inverse(left_wrist_parent_quat), body_global_rotation[20])

        left_elbow_roll, left_wrist_pitch, left_wrist_yaw = quat_in_xyz_axis(left_wrist_local_quat, seq='XYZ')
        robot_local_rotation[16] = left_elbow_roll
        robot_local_rotation[17] = left_wrist_pitch
        robot_local_rotation[18] = left_wrist_yaw

        # cal right wrist
        right_wrist_parent_quat = quat_mul_norm(
            body_global_rotation[13],
            quat_mul_four(
                robot_local_rotation[21],
                robot_local_rotation[22],
                robot_local_rotation[23],
                robot_local_rotation[24],
            )
        )

        right_wrist_local_quat = quat_mul_norm(quat_inverse(right_wrist_parent_quat), body_global_rotation[16])
        right_elbow_roll, right_wrist_pitch, right_wrist_yaw = quat_in_xyz_axis(right_wrist_local_quat, seq='XYZ')
        robot_local_rotation[25] = right_elbow_roll
        robot_local_rotation[26] = right_wrist_pitch
        robot_local_rotation[27] = right_wrist_yaw

        return robot_local_rotation

    def _retarget_wrist_from_global_translation(self, source_global_translation):
        pass


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

    df = pd.read_csv('test_motion/mocap_raw/test1.csv')
    motion_global_translation = to_torch(get_vtrdyn_full_translation(df))
    motion_global_rotation = to_torch(get_vtrdyn_full_rotation(df))
    motion_global_rotation = vtrdyn_full_zero_pose_transform(global_rotation=motion_global_rotation)

    mocap_data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
            zip(motion_global_translation, motion_global_rotation)]
    # vis_robot([mocap_data],[vtrdyn_full_zero_pose])

    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')
    vis_zero_pose([hu_zero_pose])

    hu_retarget = FullBodyRetargeter(vtrdyn_full_zero_pose, hu_zero_pose)

    for i in range(1000):
        start = time.time()
        global_translation = motion_global_translation[i]
        body_global_translation = global_translation[
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]]
        left_hand_global_translation = global_translation[14:14 + 19]
        right_hand_global_translation = global_translation[39:39 + 19]

        global_rotation = motion_global_rotation[i]
        body_global_rotation = global_rotation[
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]]
        left_hand_global_rotation = global_rotation[14:14 + 19]
        right_hand_global_rotation = global_rotation[39:39 + 19]

        hu_retarget.retarget(
            body_global_rotation,
            body_global_translation,
            left_hand_global_rotation,
            left_hand_global_translation,
            right_hand_global_rotation,
            right_hand_global_translation
        )

        end = time.time()
        print(f'Time cost {end - start:.5f} s')

    robot_data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
                  zip(hu_retarget.motion_global_translation, hu_retarget.motion_global_rotation)]

    vis_robots([robot_data, mocap_data], [hu_zero_pose, vtrdyn_full_zero_pose])
