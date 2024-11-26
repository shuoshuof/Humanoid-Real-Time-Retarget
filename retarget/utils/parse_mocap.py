# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 12:00
@Auth ： shuoshuof
@File ：parse_mocap.py
@Project ：Humanoid-Real-Time-Retarget
"""
import pandas as pd
import numpy as np

from retarget.robot_config.VTRDYN import VTRDYN_JOINT_NAMES

def get_mocap_translation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_JOINT_NAMES), 3))
    for joint_idx, joint_name in enumerate(VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} position X(m)']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} position Y(m)']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} position Z(m)']
    return motion_global_translation

def get_mocap_rotation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_JOINT_NAMES), 4))
    for joint_idx, joint_name in enumerate(VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} quaternion X']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} quaternion Y']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} quaternion Z']
        motion_global_translation[:, joint_idx, 3] = data[f'{joint_name} quaternion W']
    return motion_global_translation

if __name__ == '__main__':
    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    # get_motion_quaternions(df)
    motion_global_translation = get_mocap_translation(df)
    motion_global_rotation = get_mocap_rotation(df)


