# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/20 11:54
@Auth ： shuoshuof
@File ：sim_teleop.py
@Project ：Humanoid-Real-Time-Retarget
"""
from isaacgym import gymtorch
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import time
from sim.env import Env

import pickle
import torch

from retarget.robot_kinematics_model import RobotZeroPose

from retarget.utils import get_mocap_translation
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch, to_numpy

from retarget.robot_config.Hu_v5 import Hu_DOF_AXIS

from retarget.retarget_solver import HuUpperBodyFromMocapRetarget


cam_img = None

class MocapControlEnv(Env):
    def __init__(self, print_freq=False):
        super().__init__(print_freq)

    def _set_humanoid_dof_pos(self,env_idx,dof_pos):
        env_handle = self.env_handles[env_idx]
        actor_handle = self.robot_handles[env_idx]

        # self.gym.set_actor_dof_state(self.sim, actor_handle, to_torch(dof_pos))
        self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(dof_pos) )


    def step(self,dof_pos):
        self._set_humanoid_dof_pos(0,dof_pos)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)



class MocapReceiver:
    def 










if __name__ == '__main__':

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = get_mocap_translation(df)
    motion_global_translation = to_torch(motion_global_translation)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    hu_retarget = HuUpperBodyFromMocapRetarget(vtrdyn_zero_pose,hu_zero_pose)
    mocap_control_env = MocapControlEnv()

    for i in range(300):
        _, dof_pos = hu_retarget.retarget_from_global_translation(motion_global_translation[i])
        mocap_control_env.step(dof_pos)

