# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 13:34
@Auth ： shuoshuof
@File ：base_robot.py
@Project ：Humanoid-Real-Time-Retarget
"""
import copy
from typing import Dict,Union,OrderedDict
from collections import OrderedDict
import torch

from retarget.utils import parse_urdf
from poselib.poselib.skeleton.skeleton3d import SkeletonState

from robot_kinematics_model.base_forward_model import cal_forward_kinematics

class BaseRobot:
    def __init__(self):
        pass


class RobotZeroPose:
    def __init__(
            self,
            local_translation,
            global_translation,
            parent_indices,
            num_joints,
            node_names,
            skeleton_tree
    ):
        self._local_translation = local_translation
        self._global_translation = global_translation
        self._parent_indices = parent_indices
        self._num_joints = num_joints
        self._node_names = node_names

        self._global_rotation = torch.tensor([[0,0,0,1.]]*(self.num_joints),dtype=torch.float32)
        self._local_rotation = torch.tensor([[0,0,0,1.]]*(self.num_joints),dtype=torch.float32)

        self._skeleton_tree = skeleton_tree
    @property
    def local_translation(self):
        return self._local_translation.clone()
    @property
    def global_translation(self):
        return self._global_translation.clone()
    @property
    def global_rotation(self):
        return self._global_rotation.clone()
    @property
    def local_rotation(self):
        return self._local_rotation.clone()
    @property
    def parent_indices(self):
        return self._parent_indices.clone()
    @property
    def num_joints(self):
        return self._num_joints
    @property
    def node_names(self):
        return self._node_names
    @property
    def skeleton_tree(self):
        return copy.deepcopy(self._skeleton_tree)
    @classmethod
    def from_urdf(cls, urdf_path):
        robot_zero_pose, link_mesh_file_names = parse_urdf(urdf_path)
        return cls(
            local_translation=robot_zero_pose.local_translation,
            global_translation=robot_zero_pose.global_translation,
            parent_indices=robot_zero_pose.skeleton_tree.parent_indices,
            num_joints=robot_zero_pose.skeleton_tree.num_joints,
            node_names=robot_zero_pose.skeleton_tree.node_names,
            skeleton_tree=robot_zero_pose.skeleton_tree
        )
    @classmethod
    def from_skeleton_state(cls, skeleton_state:SkeletonState):
        return cls(
            local_translation=skeleton_state.local_translation,
            global_translation=skeleton_state.global_translation,
            parent_indices=skeleton_state.skeleton_tree.parent_indices,
            num_joints=skeleton_state.skeleton_tree.num_joints,
            node_names=skeleton_state.skeleton_tree.node_names,
            skeleton_tree=skeleton_state.skeleton_tree
        )
    @classmethod
    def from_dict(cls, robot_dict:Union[Dict,OrderedDict],is_local=False):
        if is_local:
            robot_dict['global_translation'] = cls.cal_global_translation(robot_dict['local_translation'],robot_dict['parent_indices'])
        else:
            robot_dict['local_translation'] = cls.cal_local_translation(robot_dict['global_translation'],robot_dict['parent_indices'])
        return cls(**robot_dict)
    @staticmethod
    def cal_local_translation(global_translation, parent_indices):
        local_translation = global_translation.clone()
        local_translation[1:] -= global_translation[parent_indices[1:]]
        return local_translation
    @staticmethod
    def cal_global_translation(local_translation, parent_indices):
        raise NotImplementedError
    def rebuild_pose_by_local_rotation(self,local_rotation):
        global_rotation, self._global_translation = cal_forward_kinematics(
            motion_local_rotation=local_rotation,
            motion_root_translation=self.global_translation[0],
            parent_indices=self.parent_indices,
            zero_pose_local_translation=self.local_translation
        )
        self._local_translation = self.cal_local_translation(self.global_translation,self.parent_indices)
        return global_rotation
