# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/21 15:34
@Auth ： shuoshuof
@File ：vedo_skeleton.py
@Project ：Humanoid-Real-Time-Retarget
"""

from abc import ABC,abstractmethod
import copy

import numpy as np
from vedo import *
import torch


from poselib.poselib.core.rotation3d import *

from robot_kinematics_model import RobotZeroPose
from vedo_visualizer.vedo_geoms import *
from retarget.torch_ext import *

class BaseRobot(ABC):
    def __init__(self):
        self._orig_geoms = []
        self._geoms = []
    @property
    def geoms(self):
        return copy.deepcopy(self._geoms)
    @property
    def orig_geoms(self):
        return copy.deepcopy(self._orig_geoms)
    @abstractmethod
    def _generate_geoms(self, *args, **kwargs):
        pass
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    def show(self):
        plotter = Plotter(axes=1, bg='white')
        plotter.show(*self.geoms)


class BaseSkeletonRobot(BaseRobot):
    def __init__(self,parent_indices,node_names,num_joints):
        super().__init__()
        self._parent_indices = parent_indices
        self._node_names = node_names
        self._num_joints = num_joints
    @property
    def parent_indices(self):
        return self._parent_indices
    @property
    def node_names(self):
        return self._node_names
    @property
    def num_joints(self):
        return self._num_joints
    @classmethod
    def from_zero_pose(cls,zero_pose:RobotZeroPose):
        robot = cls(zero_pose.parent_indices, zero_pose.node_names, zero_pose.num_joints)
        # robot._generate_geoms(zero_pose.global_translation)
        return robot
    def _generate_geoms(self, motion_data):
        self._geoms.clear()

        body_pos = motion_data['body_pos']
        end = body_pos[1:]
        start = body_pos[self.parent_indices[1:]]
        if motion_data.get('body_quat',None) is not None:
            body_quat = motion_data['body_quat']
            axes = torch.eye(3).unsqueeze(0).expand(len(body_pos),-1,-1)/20
            axes = quat_rotate(to_torch(body_quat).unsqueeze(1),axes)
            # rgb
            color = torch.eye(3).unsqueeze(0).expand(len(body_pos),-1,-1)

            arrow_start = np.repeat(body_pos[:,np.newaxis],3, axis=1)
            arrow_end = body_pos[:,np.newaxis] + to_numpy(axes)
            self._geoms.append(JointAxes(arrow_start.reshape(-1,3), arrow_end.reshape(-1,3), color=to_numpy(color).reshape(-1,3)))

        self._geoms.append(SkeletonLines(start, end))
        self._geoms.append(SkeletonJoints(body_pos))

    def forward(self, body_pos):
        self._generate_geoms(body_pos)

class MeshRobot(BaseRobot):
    def __init__(self,):
        super().__init__()


if __name__ == '__main__':
    import pickle
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    robot = BaseSkeletonRobot.from_zero_pose(vtrdyn_zero_pose)
    robot.show()