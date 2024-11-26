# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 11:38
@Auth ： shuoshuof
@File ：base_retargeter.py
@Project ：Humanoid-Real-Time-Retarget
"""
from abc import ABC, abstractmethod

import torch

from robot_kinematics_model import RobotZeroPose, cal_local_rotation, cal_forward_kinematics


class BaseHumanoidRetargeter(ABC):
    def __init__(self, source_zero_pose: RobotZeroPose, target_zero_pose: RobotZeroPose):
        self.source_zero_pose = source_zero_pose
        self.target_zero_pose = target_zero_pose
        self._motion_local_rotation = []
        self._motion_dof_pos = []

    @property
    def motion_global_rotation(self):
        if hasattr(self, '_motion_global_rotation') and len(self._motion_global_rotation) == self.motion_length:
            return self._motion_global_rotation.clone()
        else:
            self._motion_global_rotation, self._motion_global_translation = cal_forward_kinematics(
                motion_local_rotation=self.motion_local_rotation,
                motion_root_translation=torch.zeros((self.motion_length, 3)),
                parent_indices=self.target_zero_pose.parent_indices,
                zero_pose_local_translation=self.target_zero_pose.local_translation
            )
            return self._motion_global_rotation.clone()

    @property
    def motion_global_translation(self):
        if hasattr(self, '_motion_global_translation') and len(self._motion_global_translation) == self.motion_length:
            return self._motion_global_translation.clone()
        else:
            self._motion_global_rotation, self._motion_global_translation = cal_forward_kinematics(
                motion_local_rotation=self.motion_local_rotation,
                motion_root_translation=torch.zeros((self.motion_length, 3)),
                parent_indices=self.target_zero_pose.parent_indices,
                zero_pose_local_translation=self.target_zero_pose.local_translation
            )
            return self._motion_global_translation.clone()

    @property
    def motion_local_rotation(self):
        return torch.stack(self._motion_local_rotation).clone()

    @property
    def motion_dof_pos(self):
        return torch.stack(self._motion_dof_pos).clone()

    @property
    def motion_length(self):
        return len(self._motion_local_rotation)
