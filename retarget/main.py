# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 11:44
@Auth ： shuoshuof
@File ：main.py
@Project ：Humanoid-Real-Time-Retarget
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import time

import pickle

import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonState,SkeletonMotion
from poselib.poselib.visualization.common import plot_skeleton_H

from retarget.robot_kinematics_model import RobotZeroPose

from retarget.utils import get_mocap_translation
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch,to_numpy

from body_visualizer.common import vis_vtrdyn



class Retarget(ABC):
    def __init__(self, mocap_zero_pose:RobotZeroPose, target_zero_pose:RobotZeroPose):
        self.mocap_zero_pose = mocap_zero_pose
        self.target_zero_pose = target_zero_pose
    def cal_motion_local_rotation(self):
        pass
    @staticmethod
    def rescale_motion_to_standard_size(motion_global_translation,zero_pose:RobotZeroPose):
        rescaled_motion_global_translation = motion_global_translation.clone()
        for joint_idx, parent_idx in enumerate(zero_pose.parent_indices):
            if parent_idx == -1:
                pass
            else:
                scale = torch.linalg.norm(motion_global_translation[:, joint_idx, :] - motion_global_translation[:, parent_idx, :],dim=1) / \
                        torch.linalg.norm(zero_pose.local_translation[joint_idx, :], dim=0)
                rescaled_motion_global_translation[:, joint_idx, :] = rescaled_motion_global_translation[:,parent_idx, :] + \
                      (motion_global_translation[:, joint_idx,:] - motion_global_translation[:, parent_idx,:]) / scale.unsqueeze(1).repeat(1, 3)
        return rescaled_motion_global_translation



class RetargetHuV5fromMocap(Retarget):
    def __init__(self, mocap_zero_pose:RobotZeroPose, target_zero_pose:RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)

    def cal_shoulder_spherical_joint_rotation(self,v1, v0, parent_global_rotation):
        r"""
        calculate shoulder spherical joint rotation
        the order is pitch and roll. the raw need to use the child joint to cal
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

        pitch_joint_quat = quat_from_angle_axis(torch.tensor([theta1 - theta0]), axis[1])

        phi1 = radians_between_vecs(v1_proj, v1, n=torch.cross(v1_proj, axis[1]))
        phi0 = radians_between_vecs(v0_proj, v0, n=torch.cross(v0_proj, axis[1]))

        print(f"rotation: {theta1 - theta0}, {phi1 - phi0}")

        roll_joint_quat = quat_from_angle_axis(phi1 - phi0, axis[0])

        return pitch_joint_quat, roll_joint_quat

    def _rebuild_with_vtrdyn_zero_pose(self,motion_global_translation,fps=30)->SkeletonState:
        motion_length, num_joint, _ = motion_global_translation.shape

        rebuilt_motion_root_translation = motion_global_translation[:,0,:].clone()
        rebuilt_motion_global_rotation = torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,num_joint,1)

        zero_pose_local_translation = \
            self.mocap_zero_pose.local_translation.repeat(motion_length, 1, 1)
        parent_indices = self.mocap_zero_pose.parent_indices

        joint0_quat = cal_joint_quat(
            zero_pose_local_translation[:,[4,1,7]],
            motion_global_translation[:,[4,1,7]]-motion_global_translation[:,[0]]
        )

        joint10_quat = cal_joint_quat(
            zero_pose_local_translation[:,[17,13,11]],
            motion_global_translation[:,[17,13,11]]-motion_global_translation[:,[10]]
        )
        rebuilt_motion_global_rotation[:,[0,10],:] = \
            torch.concatenate([joint0_quat.unsqueeze(1),joint10_quat.unsqueeze(1)],dim=1)
        # joint8_quat = cal_joint_quat(
        #     zero_pose_local_translation[:,[17,13,11]],
        #     motion_global_translation[:,[17,13,11]]-motion_global_translation[:,[10]]
        # )
        # rebuilt_motion_global_rotation[:,[0,8],:] = \
        #     torch.concatenate([joint0_quat.unsqueeze(1),joint8_quat.unsqueeze(1)],dim=1)

        for joint_idx,parent_idx in enumerate(parent_indices):
            # if joint_idx == 0 or parent_idx == 0 or parent_idx==8:
            if joint_idx == 0 or parent_idx == 0 or parent_idx==10:
                continue
            else:
                rebuilt_motion_global_rotation[:,parent_idx,:] = quat_between_two_vecs(
                    zero_pose_local_translation[:,joint_idx],
                    motion_global_translation[:,joint_idx]-motion_global_translation[:,parent_idx]
                )

        rebuilt_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.mocap_zero_pose.skeleton_tree,
            rebuilt_motion_global_rotation,
            rebuilt_motion_root_translation,
            is_local=False
        )
        rebuilt_motion = SkeletonMotion.from_skeleton_state(rebuilt_skeleton_state,fps=fps)

        rebuild_error = torch.abs(rebuilt_motion.global_translation-motion_global_translation).max()
        print(f"Rebuild error :{rebuild_error:.5f}")

        return rebuilt_motion



    def retarget_from_global_translation(self, global_translation):
        motion_global_translation = coord_transform(global_translation, dir=torch.Tensor([-1, -1, 1]))

        motion_global_translation = self.rescale_motion_to_standard_size(motion_global_translation,self.mocap_zero_pose)

        mocap_motion = self._rebuild_with_vtrdyn_zero_pose(motion_global_translation)


        mocap_local_rotation = mocap_motion.local_rotation
        mocap_global_rotation = mocap_motion.global_rotation
        mocap_global_translation = mocap_motion.global_translation


        motion_length = motion_global_translation.shape[0]
        robot_local_rotation = torch.tensor([[[0, 0, 0, 1.]]*self.target_zero_pose.num_joints]*motion_length)

        robot_root_translation = torch.zeros_like(mocap_motion.root_translation)


        # root_rotation = cal_joint_quat(
        #     self.mocap_zero_pose.local_translation.unsqueeze(0)[:,[4,1,7]],
        #     motion_global_translation[:,[4,1,7]]-motion_global_translation[:,[0]]
        # )
        #
        # torso_rotation = cal_joint_quat(
        #     self.mocap_zero_pose.local_translation.unsqueeze(0),
        #     motion_global_translation[:,[11,10,15]]-motion_global_translation[:,[8]]
        # )
        #
        # torso_rotation = quat_yaw_rotation(torso_rotation)
        #

        for frame in range(motion_length):
            start = time.time()
            left_shoulder_pitch, left_shoulder_roll = self.cal_shoulder_spherical_joint_rotation(
                mocap_global_translation[frame,19]-mocap_global_translation[frame,18],
                self.mocap_zero_pose.local_translation[19],
                mocap_global_rotation[frame,10]
            )

            right_shoulder_pitch, right_shoulder_roll = self.cal_shoulder_spherical_joint_rotation(
                mocap_global_translation[frame,15]-mocap_global_translation[frame,14],
                self.mocap_zero_pose.local_translation[15],
                mocap_global_rotation[frame,10]
            )

            robot_local_rotation[frame,12] = left_shoulder_pitch
            robot_local_rotation[frame,13] = left_shoulder_roll

            robot_local_rotation[frame,21] = right_shoulder_pitch
            robot_local_rotation[frame,22] = right_shoulder_roll

            print(f'Frame {frame}: {time.time()-start:.5f} s')


        # # mocap joint 18 left shoulder
        # left_shoulder_rotation = mocap_local_rotation[:,18]
        # robot_left_shoulder_roll, robot_left_shoulder_pitch, _ = quat_in_xyz_axis(left_shoulder_rotation,'ZXY')
        #
        # robot_local_rotation[:,12] = robot_left_shoulder_pitch
        # robot_local_rotation[:,13] = robot_left_shoulder_roll
        #
        # # mocap joint 14 right shoulder
        # right_shoulder_rotation = mocap_local_rotation[:,14]
        # robot_right_shoulder_roll, robot_right_shoulder_pitch, _ = quat_in_xyz_axis(right_shoulder_rotation,'ZXY')
        #
        # robot_local_rotation[:,21] = robot_right_shoulder_pitch
        # robot_local_rotation[:,22] = robot_right_shoulder_roll

        # left_elbow_rotation = mocap_local_rotation[:,19]
        # _,left_elbow_pitch,left_shoulder_yaw = quat_in_xyz_axis(left_elbow_rotation,'ZYX')
        # robot_local_rotation[:,14] = left_shoulder_yaw
        # robot_local_rotation[:,15] = left_elbow_pitch
        #
        # right_elbow_rotation = mocap_local_rotation[:,15]
        # _,right_elbow_pitch,right_shoulder_yaw = quat_in_xyz_axis(right_elbow_rotation,'ZYX')
        # robot_local_rotation[:,23] = right_shoulder_yaw
        # robot_local_rotation[:,24] = right_elbow_pitch

        retargeted_state = SkeletonState.from_rotation_and_root_translation(
            self.target_zero_pose.skeleton_tree,
            robot_local_rotation,
            robot_root_translation,
            is_local=True
        )

        retargeted_motion = SkeletonMotion.from_skeleton_state(retargeted_state,fps=30)

        plot_skeleton_H([mocap_motion,retargeted_motion])











if __name__ == '__main__':

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = get_mocap_translation(df)
    motion_global_translation = to_torch(motion_global_translation)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)
    # plot_skeleton_H([vtrdyn_zero_pose])


    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    retargeter = RetargetHuV5fromMocap(vtrdyn_zero_pose,hu_zero_pose)
    retargeter.retarget_from_global_translation(motion_global_translation)
    # for i in range(100):
    #     start = time.time()
    #     retargeter.retarget_from_global_translation(motion_global_translation)
    #     end = time.time()
    #     print(f'Time cost {end-start:.5f} s')




