# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/29 15:05
@Auth ： shuoshuof
@File ：mujoco_env.py
@Project ：Humanoid-Real-Time-Retarget
"""
import time

import mujoco
import mujoco.viewer
import numpy as np


class MujocoTeleopEnv:
    def __init__(self,scene_path):
        self._load_scene(scene_path)
        self.set_buffers()

    def _load_scene(self,mjcf_path):
        self.scene = mujoco.MjModel.from_xml_path(mjcf_path)

    def set_buffers(self):
        self.mj_data = mujoco.MjData(self.scene)

        self.root_pos = self.mj_data.qpos[:3]
        self.root_rot = self.mj_data.qpos[3:7]

        self.root_vel = self.mj_data.qvel[:3]
        self.root_ang_vel = self.mj_data.qvel[3:6]

        self.dof_pos = self.mj_data.qpos[7:]




    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.scene,self.mj_data)

    def step(self,dof_pos):
        assert self.viewer is not None and self.viewer.is_running()
        self._set_dof_pos(dof_pos)
        self._fix_root()
        mujoco.mj_step(self.scene,self.mj_data)

        self.viewer.sync()
    def _fix_root(self):
        self.root_pos=np.array([0,0,1.1])
        self.root_rot=np.array([1,0,0,0])
        self.root_vel = np.zeros(3)
        self.root_ang_vel = np.zeros(3)
    def _set_dof_pos(self,dof_pos):
        for i in range(len(dof_pos)):
            self.dof_pos[i] = dof_pos[i]

    def run(self):
        pass

if __name__ == "__main__":
    env = MujocoTeleopEnv("asset/hu/scene.xml")
    env.launch_viewer()
    dof_pos = np.zeros(env.scene.nq-7)
    while True:
        env.step(dof_pos)










