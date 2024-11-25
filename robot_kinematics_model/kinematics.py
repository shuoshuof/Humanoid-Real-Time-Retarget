# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 11:12
@Auth ： shuoshuof
@File ：kinematics.py
@Project ：Humanoid-Real-Time-Retarget
"""
import torch

from poselib.poselib import *


def cal_forward_kinematics(motion_local_rotation, motion_root_translation, parent_indices:list, zero_pose_local_translation):
    """
    Args:
        motion_local_rotation (torch.Tensor): (L, J, 4)
        motion_root_translation (torch.Tensor): (L, 3)
        parent_indices (torch.Tensor):  (J)
        zero_pose_local_translation (torch.Tensor): (J, 3)
    Returns:
        tuple:
            - motion_global_rotation (torch.Tensor): (N, J, 4)
            - motion_global_translation (torch.Tensor): (N, J, 3)
    """
    motion_global_rotation = []
    motion_global_translation = []
    for joint_idx, parent_idx in enumerate(parent_indices):
        if parent_idx == -1:
            motion_global_rotation.append(motion_local_rotation[...,joint_idx,:])
            motion_global_translation.append(motion_root_translation)
        else:
            motion_global_rotation.append(
                quat_mul_norm(motion_global_rotation[parent_idx],motion_local_rotation[..., joint_idx, :]))
            motion_global_translation.append(quat_rotate(motion_global_rotation[parent_idx], zero_pose_local_translation[joint_idx, :])
                                             + motion_global_translation[parent_idx])

    motion_global_rotation = torch.stack(motion_global_rotation, dim=-2)
    motion_global_translation = torch.stack(motion_global_translation, dim=-2)
    return motion_global_rotation, motion_global_translation

def cal_local_rotation(motion_global_rotation, parent_indices:list):
    """
    Args:
        motion_global_rotation (torch.Tensor): (N, J, 4)
        parent_indices (torch.Tensor):  (J)
        zero_pose_local_translation (torch.Tensor): (J, 3)
    Returns:
        tuple:
            - motion_local_rotation (torch.Tensor): (N, J, 4)
            - motion_local_translation (torch.Tensor): (N, J, 3)
    """
    local_rotation = quat_identity_like(motion_global_rotation)

    for joint_idx, parent_idx in enumerate(parent_indices):
        if parent_idx == -1:
            local_rotation[...,joint_idx,:] = motion_global_rotation[...,joint_idx,:]
        else:
            local_rotation[...,joint_idx,:] = quat_mul_norm(
                quat_inverse(motion_global_rotation[...,parent_idx,:]),
                motion_global_rotation[...,joint_idx,:]
            )

    return local_rotation










