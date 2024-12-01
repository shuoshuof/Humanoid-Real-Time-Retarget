# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/19 18:01
@Auth ： shuoshuof
@File ：env.py
@Project ：Humanoid-Real-Time-Retarget
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import random
import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


class Env:
    def __init__(self,print_freq=False):
        self.print_freq = print_freq

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        self.gym = gymapi.acquire_gym()
        self.create_sim()

    def create_sim(self):
        sim_params = self.set_sim_parameters()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        self._create_ground_plane()
        self._create_envs()
        self._create_viewer()
    def set_sim_parameters(self):
        sim_params = gymapi.SimParams()

        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        return sim_params

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs=1, env_spacing=1.25, num_per_row=1):
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)

        self.env_handles = []
        self.robot_handles = []
        self.object_handles = []
        self.table_handles = []
        self.target_handles = []
        self.left_cam_handles = []
        self.right_cam_handles = []
        self.center_cam_handles = []
        # create and populate the environments
        for i in range(num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,num_per_row)
            self.env_handles.append(env_handle)

            table_handle = self._add_table(env_handle,i)
            self.table_handles.append(table_handle)

            object_handle = self._add_object(env_handle,i)
            self.object_handles.append(object_handle)

            target_handle = self._add_target(env_handle,i)
            self.target_handles.append(target_handle)

            robot_handle = self._add_robot(env_handle, i)
            self.robot_handles.append(robot_handle)

            left_camera_handle,right_camera_handle = self._create_eye_cameras(env_handle)
            self.left_cam_handles.append(left_camera_handle)
            self.right_cam_handles.append(right_camera_handle)

            center_camera_handle = self._create_center_camera(env_handle)
            self.center_cam_handles.append(center_camera_handle)

    def _add_robot(self, env, env_index):
        asset_root = "asset/hu/"
        asset_path = 'hu_v5.urdf'

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True

        asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        robot_handle = self.gym.create_actor(env, asset, pose, 'robot', env_index, 1)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)

        self.dof_dict = {value: idx
                         for (idx, value) in enumerate(self.gym.get_asset_dof_names(asset))}
        print(self.dof_dict)

        rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
        for rigid_shape_prop in rigid_shape_props:
            rigid_shape_prop.friction = 10

        self.gym.set_actor_rigid_shape_properties(env, robot_handle, rigid_shape_props)
        self._set_dof_properties(env,robot_handle)

        return robot_handle
    def _set_dof_properties(self,env,robot_handle):
        dof_props = self.gym.get_actor_dof_properties(env, robot_handle)
        for dof_kp in dof_props["stiffness"]:
            pass
        for dof_kd in dof_props["damping"]:
            dof_kd = 10
        dof_props["damping"][self.dof_dict['left_gripper_left_joint']] = 10
        dof_props["damping"][self.dof_dict['left_gripper_right_joint']] = 10

        dof_props["damping"][self.dof_dict['right_gripper_left_joint']] = 10
        dof_props["damping"][self.dof_dict['right_gripper_right_joint']] = 10

        self.gym.set_actor_dof_properties(env, robot_handle, dof_props)

    def _create_center_camera(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        center_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(center_camera_handle,
                                     env,
                                     gymapi.Vec3(*self.cam_pos),
                                     gymapi.Vec3(*self.cam_pos + self.cam_lookat_offset))
        cam_pos = gymapi.Vec3(-0.5, 0, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.set_camera_location(center_camera_handle,env,cam_pos,cam_target)
        return center_camera_handle
    def _add_table(self,env,env_index):
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.05, table_asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(env, table_asset, pose, 'table', env_index,1)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        return table_handle
    def _add_object(self, env, env_idx):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 50
        object_asset_options.thickness = 0.001

        object_asset = self.gym.create_box(self.sim, 0.05, 0.08  , 0.08, object_asset_options)

        # asset_root = "asset/teleop/"
        # asset_path = 'cylinder.urdf'
        #
        # object_asset = self.gym.load_asset(self.sim, asset_root, asset_path, object_asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.2, random.uniform(-0.15,0.05), 1.3)
        # pose.p = gymapi.Vec3(-0.2, -0.2, 1.3)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        object_handle = self.gym.create_actor(env, object_asset, pose, 'cylinder', env_idx,-1)
        color = gymapi.Vec3(1, 0., 0.)
        self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


        rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, object_handle)
        rigid_shape_props[0].friction = 20
        self.gym.set_actor_rigid_shape_properties(env, object_handle, rigid_shape_props)
        return object_handle
    def _add_target(self,env,env_idx):
        target_asset_options = gymapi.AssetOptions()
        target_asset_options.density = 1
        target_asset_options.disable_gravity = True
        target_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.00001, target_asset_options)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.1, 0, 1.125001)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        target_handle = self.gym.create_actor(env, target_asset, pose, 'target', env_idx+1,0)
        color = gymapi.Vec3(0., 1, 0.)
        self.gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        return target_handle
    def _create_eye_cameras(self,env):
        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        left_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(left_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        right_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(right_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

        return left_camera_handle,right_camera_handle
    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(-0.5, 0, 1.7)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        # self.viewer_cam_handle = self.gym.get_viewer_camera_handle(self.viewer)

    def get_eye_cameras_image(self,env_idx,show=False):
        env = self.env_handles[env_idx]
        left_camera_handle = self.left_cam_handles[env_idx]
        right_camera_handle = self.right_cam_handles[env_idx]
        left_image = self.gym.get_camera_image(self.sim,env, left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim,env, right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]
        if show:
            img = np.concatenate([left_image,right_image],axis=1)
            plt.imshow(img)
            plt.axis('off')
            plt.pause(0.001)
        return left_image,right_image
    def run(self):

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            self.get_eye_cameras_image(env_idx=0,show=False)

            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

if __name__ == '__main__':
    env = Env()
    env.run()