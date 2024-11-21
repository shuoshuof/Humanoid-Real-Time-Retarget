# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/20 11:54
@Auth ： shuoshuof
@File ：sim_teleop.py
@Project ：Humanoid-Real-Time-Retarget
"""
from isaacgym import gymtorch,gymapi
from abc import ABC, abstractmethod
import threading
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
from mocap_communication.receive import MocapReceiver


class MocapControlEnv(Env):
    def __init__(self, print_freq=False):
        super().__init__(print_freq)

    def _set_humanoid_dof_pos(self,env_idx,dof_pos):
        env_handle = self.env_handles[env_idx]
        actor_handle = self.robot_handles[env_idx]

        # self.gym.set_actor_dof_state(self.sim, actor_handle, to_torch(dof_pos))
        self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(dof_pos) )
    def _get_viewer_img(self,env_dix):
        env_handle = self.env_handles[env_dix]
        cam_handle = self.center_cam_handles[env_dix]
        img = self.gym.get_camera_image(self.sim,env_handle, cam_handle, gymapi.IMAGE_COLOR)
        return img.reshape(img.shape[0],-1,4)[...,:3]

    def _get_dof_states(self):
        pass

    def step(self,dof_pos):
        self._set_humanoid_dof_pos(0,dof_pos)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        viewer_img = self._get_viewer_img(0)

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        print(viewer_img.shape)
        return viewer_img

class DataRecorder:
    def __init__(self,save_dir):
        self.body_pose = []
        self.dof_pos = []
        self.img = []
        self.save_dir = save_dir

    def record(self,body_pose,dof_pos,img):
        assert body_pose.shape == (23,3)
        self.body_pose.append(to_numpy(body_pose))
        self.dof_pos.append(to_numpy(dof_pos))
        self.img.append(img)

    def save(self):
        data = {'body_pose':np.array(self.body_pose),'dof_pos':np.array(self.dof_pos)}
        np.savez(self.save_dir,**data)










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

    # receiver = MocapReceiver('192.168.1.167', 12345)
    #
    # receiver_thread = threading.Thread(target=receiver.run)
    # receiver_thread.start()
    # if receiver.has_connected.wait(timeout=20):
    if True:
        last_body_pose = torch.zeros(31-1)
        while True:
            # body_pose = receiver.get_body_pose()
            body_pose = None
            if body_pose is not None:
                body_pose = to_torch(body_pose)
                print(body_pose)
                _, dof_pos = hu_retarget.retarget_from_global_translation(body_pose)
                last_body_pose = body_pose
            else:
                dof_pos = last_body_pose
            # _, dof_pos = hu_retarget.retarget_from_global_translation(motion_global_translation[i])

            mocap_control_env.step(dof_pos)

    receiver.stop()
    receiver_thread.join()
