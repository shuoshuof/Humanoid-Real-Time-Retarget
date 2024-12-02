# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/29 15:05
@Auth ： shuoshuof
@File ：mujoco_teleop_env.py
@Project ：Humanoid-Real-Time-Retarget
"""
import mujoco
import mujoco.viewer
import numpy as np
import glfw
import cv2

from sim.dof_cfg import Hu_DOF_CTRL
class MujocoTeleopEnv:
    def __init__(self, scene_path):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        self._init_sim(scene_path)
        self.kp = np.array(Hu_DOF_CTRL['kp'])
        self.kd = np.array(Hu_DOF_CTRL['kd'])

    def _init_sim(self, mjcf_path):
        self.scene = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.scene)

        self.opt = mujoco.MjvOption()
        self.scene_state = mujoco.MjvScene(self.scene, maxgeom=10000)

        # Create a hidden window to initialize OpenGL context
        self.window = glfw.create_window(640, 640, "Hidden Window", None, None)
        glfw.make_context_current(self.window)

        self.context = mujoco.MjrContext(self.scene, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._init_eye_cam()

    def _init_eye_cam(self):
        self.eye_cam = mujoco.MjvCamera()

        self.eye_cam.lookat = [0.5, 0., 0.5]
        self.eye_cam.elevation = -90
        self.eye_cam.azimuth = 180
        self.eye_cam.fixedcamid = mujoco.mj_name2id(self.scene, mujoco.mjtObj.mjOBJ_CAMERA, 'top')

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.scene, self.mj_data,key_callback=self.key_callback)
        self.viewer.cam.lookat = [0.5, 0., 0.5]
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0

    def get_camera_image(self, width=640, height=480):
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.scene, self.mj_data, self.opt, None, self.eye_cam, mujoco.mjtCatBit.mjCAT_ALL,
                               self.scene_state)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(viewport, self.scene_state, self.context)
        mujoco.mjr_readPixels(img, None, viewport, self.context)
        img = cv2.flip(img, 1)
        return img

    def _fix_root(self):
        self.mj_data.qpos[:3] = [0, 0, 0.97]
        self.mj_data.qpos[3:7] = [1, 0, 0, 0]
        self.mj_data.qvel[:3] = [0, 0, 0]
        self.mj_data.qvel[3:6] = [0, 0, 0]
        self.mj_data.qfrc_applied[:6] = 0

    def _set_dof_pos(self, dof_pos):
        limb_indices = np.array([1,2,3,4,5,
                                 6,7,8,9,10,
                                 11,
                                 12,13,14,15,16,17,18,
                                 21,22,23,24,25,26,27,])-1
        self.mj_data.qpos[limb_indices+7] = dof_pos[limb_indices]

    def _ctrl_gripper(self, dof_pos):
        #left gripper
        dof_state = self.mj_data.qpos[7:7+30].copy()
        dof_vel = self.mj_data.qvel[7:7+30].copy()

        ctrl_torque = (dof_pos - dof_state) *self.kp - dof_vel * self.kd

        self.mj_data.ctrl[19 - 1] = ctrl_torque[19-1]
        self.mj_data.ctrl[20 - 1] = ctrl_torque[20-1]
        # right gripper
        self.mj_data.ctrl[28 - 1] = ctrl_torque[28-1]
        self.mj_data.ctrl[29 - 1] = ctrl_torque[29-1]

    def step(self, dof_pos):
        assert self.viewer is not None and self.viewer.is_running()

        self._set_dof_pos(dof_pos)
        self._ctrl_gripper(dof_pos)
        self._fix_root()
        mujoco.mj_step(self.scene, self.mj_data)
        self.viewer.sync()
    def reset_simulation(self):
        # Reset the position and velocity of the object
        object_id = mujoco.mj_name2id(self.scene,mujoco.mjtObj.mjOBJ_BODY,"object")
        start_idx = self.scene.body_jntadr[object_id]+7

        self.mj_data.qpos[start_idx:start_idx+3]= np.array([0,0,3])
        self.mj_data.qvel[start_idx+3:start_idx+7]= np.array([1,0,0,0])
        self._fix_root()

    def key_callback(self, keycode):
        if chr(keycode) == 'R':
            self.reset_simulation()


if __name__ == "__main__":
    env = MujocoTeleopEnv("asset/hu/scene.xml")
    env.launch_viewer()
    dof_pos = np.zeros(30)
    while True:
        env.step(dof_pos)
        img = env.get_camera_image()
        cv2.imshow("MuJoCo Camera Image", img[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
