# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 11:47
@Auth ： shuoshuof
@File ：mocap_env.py
@Project ：Humanoid-Real-Time-Retarget
"""
from isaacgym import gymtorch, gymapi

from sim.env import Env


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
