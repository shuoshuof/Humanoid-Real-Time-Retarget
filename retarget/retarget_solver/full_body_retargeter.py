# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 14:13
@Auth ： shuoshuof
@File ：hand_retargeter.py
@Project ：Humanoid-Real-Time-Retarget
"""
from retarget.retarget_solver.base_retargeter import BaseHumanoidRetargeter
from robot_kinematics_model import RobotZeroPose, cal_local_rotation

class FullBodyRetargeter(BaseHumanoidRetargeter):
    def __init__(self, mocap_zero_pose: RobotZeroPose, target_zero_pose: RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)

    def retarget(self,source_global_rotation, source_global_translation):
        pass

    def retarget_arm_from_global_translation(self,source_global_translation):
        pass

    def retarget_arm_from_global_rotation(self,source_global_rotation):
        pass
    def retarget_hand_from_global_rotation(self,source_global_rotation):
        pass

    def retarget_hand_from_global_translation(self,source_global_translation):
        pass


if __name__ == '__main__':
    import pickle
    import pandas as pd

    from robot_kinematics_model import RobotZeroPose
    from vedo_visualizer.common import vis_zero_pose,vis_robot

    from retarget.torch_ext import to_torch
    from retarget.utils.parse_mocap import get_vtrdyn_full_translation,get_vtrdyn_full_rotation
    from retarget.utils.parse_mocap import vtrdyn_full_zero_pose_transform

    with open('asset/zero_pose/vtrdyn_full_zero_pose.pkl','rb') as f:
        vtrdyn_full_zero_pose = pickle.load(f)

    vtrdyn_full_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_full_zero_pose)


    df = pd.read_csv('test_motion/mocap_raw/test1.csv')
    motion_global_translation = to_torch(get_vtrdyn_full_translation(df))
    motion_global_rotation = to_torch(get_vtrdyn_full_rotation(df))
    motion_global_rotation = vtrdyn_full_zero_pose_transform(global_rotation=motion_global_rotation)

    data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
            zip(motion_global_translation, motion_global_rotation)]

    vis_robot([data],vtrdyn_full_zero_pose)





    vis_zero_pose([vtrdyn_full_zero_pose])

