# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 13:34
@Auth ： shuoshuof
@File ：base_robot.py
@Project ：Humanoid-Real-Time-Retarget
"""
import copy

from retarget.utils import parse_urdf
from poselib.poselib.skeleton.skeleton3d import SkeletonState


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
        self._skeleton_tree = skeleton_tree
    @property
    def local_translation(self):
        return self._local_translation.clone()
    @property
    def global_translation(self):
        return self._global_translation.clone()
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




