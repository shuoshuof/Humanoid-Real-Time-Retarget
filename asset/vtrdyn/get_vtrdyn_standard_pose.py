from bvh import Bvh
import numpy as np
import pickle
import copy
from collections import OrderedDict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.core.rotation3d import *
from retarget.robot_config.VTRDYN import VTRDYN_JOINT_NAMES,vtrdyn_parent_indices

from retarget.spatial_transform.transform3d import coord_transform

from robot_kinematics_model import RobotZeroPose
from vedo_visualizer.common import vis_zero_pose
if __name__ == "__main__":

    with open('asset/vtrdyn/vtrdyn_t_pose.bvh') as f:
        vtrdyn_data = Bvh(f.read())

    t_pose_local_translation = []
    vtrdyn_joints_names = VTRDYN_JOINT_NAMES
    for joint_name in vtrdyn_joints_names:
        t_pose_local_translation.append(vtrdyn_data.joint_offset(joint_name))
    t_pose_local_translation = np.array(t_pose_local_translation)/100

    t_pose_local_translation = coord_transform(t_pose_local_translation,order=[2,0,1],dir=np.array([1,-1,1]))

    vtrdyn_sk_tree = SkeletonTree.from_dict(
        OrderedDict(
            node_names=np.array(VTRDYN_JOINT_NAMES),
            parent_indices={'arr': np.array(vtrdyn_parent_indices), 'context': {'dtype': 'int64'}},
            local_translation={'arr': t_pose_local_translation.copy(), 'context': {'dtype': 'float32'}}
        )
    )


    vtrdyn_t_pose = SkeletonState.zero_pose(vtrdyn_sk_tree)
    vis_zero_pose([RobotZeroPose.from_skeleton_state(vtrdyn_t_pose)])

    with open('asset/t_pose/vtrdyn_t_pose.pkl', 'wb') as f:
        pickle.dump(vtrdyn_t_pose, f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_t_pose)

    zero_pose_local_rotation = vtrdyn_zero_pose.local_rotation.clone()
    zero_pose_local_rotation[18] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([1., 0., 0.]))
    zero_pose_local_rotation[19] = quat_from_angle_axis(torch.tensor(-torch.pi / 2), torch.tensor([0., 0, 1.]))
    zero_pose_local_rotation[14] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([1., 0., 0.]))
    zero_pose_local_rotation[15] = quat_from_angle_axis(torch.tensor(torch.pi / 2), torch.tensor([0., 0., 1.]))
    vtrdyn_zero_pose.rebuild_pose_by_local_rotation(zero_pose_local_rotation)
    sk_state = vtrdyn_zero_pose.get_sk_zero_pose()


    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'wb') as f:
        pickle.dump(sk_state, f)


    vis_zero_pose([vtrdyn_zero_pose])





