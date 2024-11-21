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
from collections import OrderedDict
import numpy as np
import time
import pickle
import cv2

from sim.env import Env
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

    def _set_humanoid_dof_tar_pos(self, env_idx, dof_pos):
        env_handle = self.env_handles[env_idx]
        actor_handle = self.robot_handles[env_idx]

        # self.gym.set_actor_dof_state(self.sim, actor_handle, to_torch(dof_pos))
        self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(dof_pos) )
    def _get_viewer_img(self,env_dix):
        env_handle = self.env_handles[env_dix]
        cam_handle = self.center_cam_handles[env_dix]
        img = self.gym.get_camera_image(self.sim,env_handle, cam_handle, gymapi.IMAGE_COLOR)
        return img.reshape(img.shape[0],-1,4)[...,:3]

    def _get_dof_states(self,env_idx):
        env_handle = self.env_handles[env_idx]
        actor_handle = self.robot_handles[env_idx]
        dof_state = self.gym.get_actor_dof_states(env_handle,actor_handle,gymapi.STATE_POS)
        return dof_state

    def step(self, dof_tar_pos):
        self._set_humanoid_dof_tar_pos(0, dof_tar_pos)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        viewer_img = self._get_viewer_img(0)
        dof_state = self._get_dof_states(0)
        # print(dof_state)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        return dof_state, viewer_img

class DataRecorder:
    def __init__(self,save_dir):
        self.body_pose = []
        self.dof_pos = []
        self.dof_state = []
        self.img = []
        self.save_dir = save_dir

    def record(self,body_pose,dof_pos,dof_state,img):
        if body_pose is not None:
            self.body_pose.append(to_numpy(body_pose))
        self.dof_pos.append(to_numpy(dof_pos).astype(np.float32))
        self.dof_state.append(self.process_dof_state(dof_state))
        self.img.append(self.process_img(img))

    def process_img(self,img):
        assert img.shape == (720,1280,3)
        img = img[:,180:-180]
        img = cv2.resize(img.astype(np.uint8),(224,224))
        return img
    def process_dof_state(self,dof_state):
        dof_state = dof_state['pos']
        return dof_state.astype(np.float32)

    def save(self):
        # with open(f'{self.save_dir}/body_pose.pkl','wb') as f:
        #     pickle.dump(np.stack(self.body_pose),f)
        # with open(f'{self.save_dir}/dof_pos.pkl','wb') as f:
        #     pickle.dump(np.stack(self.dof_pos),f)
        # with open(f'{self.save_dir}/dof_state.pkl','wb') as f:
        #     pickle.dump(np.stack(self.dof_state),f)
        # with open(f'{self.save_dir}/img.pkl','wb') as f:
        #     pickle.dump(np.stack(self.img),f)
        data_dict = OrderedDict(
            body_pose=np.stack(self.body_pose),
            dof_pos=np.stack(self.dof_pos),
            dof_state=np.stack(self.dof_state),
            img=np.stack(self.img)
        )

        save_time = time.strftime("%Y-%m-%d_%H:%M:%S")

        with open(f'{self.save_dir}/{save_time}.pkl','wb') as f:
            pickle.dump(data_dict,f)







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

    recorder = DataRecorder('data')
    receiver = MocapReceiver('192.168.1.167', 12345)

    receiver_thread = threading.Thread(target=receiver.run)
    receiver_thread.start()
    if receiver.has_connected.wait(timeout=20):
    # if True:
        last_dof_pose = torch.zeros(31 - 1)
        while True:
            start = time.time()
            if not receiver.is_connected():
                break

            body_pose = receiver.get_body_pose()

            if body_pose is not None:
                body_pose = to_torch(body_pose)
                body_pose = body_pose[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
                _, dof_pos = hu_retarget.retarget_from_global_translation(body_pose)
                last_dof_pose = dof_pos
            else:
                dof_pos = last_dof_pose
            # _, dof_pos = hu_retarget.retarget_from_global_translation(motion_global_translation[i])
            dof_state, viewer_img = mocap_control_env.step(dof_pos)
            recorder.record(body_pose,dof_pos,dof_state,viewer_img)
            end = time.time()
            # print(f"{end - start:.3f}")

    receiver.stop()
    receiver_thread.join()

    recorder.save()


