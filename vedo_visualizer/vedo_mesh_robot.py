# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/29 11:39
@Auth ： shuoshuof
@File ：vedo_mesh_robot.py
@Project ：Humanoid-Real-Time-Retarget
"""
import copy
from typing import List
from vedo import *

from poselib.poselib.core.rotation3d import *

from retarget.utils.parse_urdf import parse_urdf
from retarget.torch_ext import *

from robot_kinematics_model.base_robot import RobotZeroPose
from vedo_visualizer.vedo_skeleton_robot import BaseRobot

class VedoMeshRobot(BaseRobot):
    def __init__(self,parent_indices,node_names,num_joints):
        super().__init__()
        self._parent_indices = parent_indices
        self._node_names = node_names
        self._num_joints = num_joints
        self._meshes:List[Mesh] = None
    @property
    def parent_indices(self):
        return self._parent_indices
    @property
    def node_names(self):
        return self._node_names
    @property
    def num_joints(self):
        return self._num_joints
    @property
    def meshes(self):
        return copy.deepcopy(self._meshes)
    @classmethod
    def from_urdf(cls,urdf_path):
        robot_zero_pose = RobotZeroPose.from_urdf(urdf_path)
        vedo_robot =  cls(robot_zero_pose.parent_indices,robot_zero_pose.node_names,robot_zero_pose.num_joints)
        vedo_robot._load_mesh(urdf_path)
        return vedo_robot
    def _load_mesh(self,urdf_path):
        _,link_mesh_file_names = parse_urdf(urdf_path)
        for link_mesh_name in link_mesh_file_names:
            mesh_path  = os.path.join(os.path.dirname(urdf_path),link_mesh_name)
            mesh = Mesh(mesh_path,alpha=0.3)
            self._meshes.append(mesh)

    def _generate_geoms(self, motion_data):
        self._geoms.clear()

        body_pos = motion_data['body_pos']
        body_quat = motion_data['body_quat']

        meshes = self.meshes

        for mesh,pos,quat in zip(meshes,body_pos,body_quat):
            angle,axis  = quat_to_angle_axis(quat)
            angle = to_numpy(angle)
            axis = to_numpy(axis)

            mesh.rotate(angle = angle,axis = axis)
            mesh.pos(pos)

            self._geoms.append(mesh)

    def forward(self, motion_data):
        self._generate_geoms(motion_data)

if __name__ == '__main__':
    pass




