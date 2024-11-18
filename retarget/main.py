# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 11:44
@Auth ： shuoshuof
@File ：main.py
@Project ：Humanoid-Real-Time-Retarget
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

import pickle

from retarget.robot_kinematics_model import RobotZeroPose

from retarget.utils import get_mocap_translation
from spatial_transform.transform3d import *



class BaseRetargetSolver(ABC):
    def __init__(self, mocap_zero_pose:RobotZeroPose, target_zero_pose:RobotZeroPose):
        self.mocap_zero_pose = mocap_zero_pose
        self.target_zero_pose = target_zero_pose

    def retarget_from_global_translation(self, global_translation):


        pass


class HuRetargetSolver(BaseRetargetSolver):
    def __init__(self, mocap_zero_pose:RobotZeroPose, target_zero_pose:RobotZeroPose):
        super().__init__(mocap_zero_pose, target_zero_pose)

    def retarget_from_global_translation(self, global_translation):
        pass









if __name__ == '__main__':

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = get_mocap_translation(df)


    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)



    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')






