# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/22 16:28
@Auth ： shuoshuof
@File ：zero_pose_transform.py
@Project ：Humanoid-Real-Time-Retarget
"""
import pickle
from collections import OrderedDict
from poselib.poselib.skeleton.skeleton3d import SkeletonState,SkeletonMotion
from poselib.poselib.core.rotation3d import *

from vedo_visualizer.common import vis_zero_pose

from robot_kinematics_model import RobotZeroPose

with open('asset/t_pose/vtrdyn_t_pose.pkl', 'rb') as f:
    vtrdyn_t_pose: SkeletonState = pickle.load(f)

vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(
    skeleton_state=vtrdyn_t_pose)

zero_pose_local_rotation = vtrdyn_zero_pose.local_rotation.clone()
zero_pose_local_rotation[18] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[19] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([0., 0, 1.]))
zero_pose_local_rotation[14] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([1., 0., 0.]))
zero_pose_local_rotation[15] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1.]))

# t pose to zero pose transform
vtrdyn_t2zero_pose_transform_quat = vtrdyn_zero_pose.rebuild_pose_by_local_rotation(zero_pose_local_rotation)

# vis_zero_pose([vtrdyn_zero_pose])

def zero_pose_transform_quat(global_rotation):
    rotation = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1]))
    transformed_global_rotation = global_rotation

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation, rotation)

    transformed_global_rotation = quat_mul_norm(transformed_global_rotation,quat_inverse(vtrdyn_t2zero_pose_transform_quat))

    return transformed_global_rotation


if __name__ == '__main__':
    from poselib.poselib.core.rotation3d import *
    from poselib.poselib.visualization.common import plot_skeleton_H
    from robot_kinematics_model.kinematics import cal_local_rotation
    from retarget.spatial_transform.transform3d import *
    from retarget.utils import get_mocap_rotation, get_mocap_translation
    from retarget.torch_ext import *
    import pandas as pd
    import numpy as np
    from vedo_visualizer.common import vis_robot

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')

    motion_global_translation = get_mocap_translation(df)
    motion_global_rotation = get_mocap_rotation(df)
    motion_global_translation = motion_global_translation - motion_global_translation[0, 0]

    motion_global_rotation = zero_pose_transform_quat(to_torch(motion_global_rotation))

    data = [{'body_pos': body_pos, 'body_quat': body_quat} for body_pos, body_quat in
            zip(motion_global_translation, motion_global_rotation)]

    # vis_mocap_robot([data])
    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_sk_zero_pose: SkeletonState = pickle.load(f)

    local_rotation = cal_local_rotation(to_torch(motion_global_rotation),vtrdyn_zero_pose.parent_indices)

    new_state = SkeletonState.from_rotation_and_root_translation(
        vtrdyn_sk_zero_pose.skeleton_tree,
        local_rotation,
        to_torch(motion_global_translation[:,0, :]),
        is_local=True
    )


    new_motion = SkeletonMotion.from_skeleton_state(new_state,fps=30)

    plot_skeleton_H([new_motion])






