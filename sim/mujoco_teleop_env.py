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

class MujocoTeleopEnv:
    def __init__(self, scene_path):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        self._init_sim(scene_path)

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
        self._init_viewer_cam()

    def _init_eye_cam(self):
        self.eye_cam = mujoco.MjvCamera()
        self.eye_cam.fixedcamid = mujoco.mj_name2id(self.scene, mujoco.mjtObj.mjOBJ_CAMERA, 'top')
        self.eye_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def _init_viewer_cam(self):
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.5, 0, 0.5]  # Focus point of the camera
        self.cam.azimuth = 45  # Horizontal rotation (degrees)
        self.cam.elevation = -30  # Vertical rotation (degrees)
        self.cam.distance = 2.0  # Distance from the focus point
    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.scene, self.mj_data)

    def step(self, dof_pos):
        assert self.viewer is not None and self.viewer.is_running()
        self._set_dof_pos(dof_pos)
        self._fix_root()
        mujoco.mj_step(self.scene, self.mj_data)
        self.viewer.sync()

    def _fix_root(self):
        self.mj_data.qpos[:3] = [0, 0, 1.]
        self.mj_data.qpos[3:7] = [1, 0, 0, 0]
        self.mj_data.qvel[:3] = [0, 0, 0]
        self.mj_data.qvel[3:6] = [0, 0, 0]
        self.mj_data.qfrc_applied[:6] = 0

    def _set_dof_pos(self, dof_pos):
        self.mj_data.qpos[7:7 + len(dof_pos)] = dof_pos

    def get_camera_image(self, width=640, height=480):
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.scene, self.mj_data, self.opt, None, self.eye_cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene_state)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(viewport, self.scene_state, self.context)
        mujoco.mjr_readPixels(img, None, viewport, self.context)
        return img

if __name__ == "__main__":
    env = MujocoTeleopEnv("asset/hu/scene.xml")
    env.launch_viewer()
    dof_pos = np.zeros(env.scene.nq - 7)
    while True:
        env.step(dof_pos)
        img = env.get_camera_image()
        img = cv2.flip(img, 0)  # Flip the image vertically
        cv2.imshow("MuJoCo Camera Image", img[...,::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()












