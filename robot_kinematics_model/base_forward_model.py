from abc import ABC
from poselib.poselib.core.rotation3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from robot_kinematics_model import cal_forward_kinematics


class BaseForwardModel(ABC):
    def __init__(self,skeleton_tree:SkeletonTree,device='cuda:0'):
        self.sk_local_translation = skeleton_tree.local_translation
        self.parent_indices = skeleton_tree.parent_indices
        self.num_joints:int = skeleton_tree.num_joints
        self.device = device
    def forward_kinematics(self,**kwargs):
        return cal_forward_kinematics(**kwargs, parent_indices=self.parent_indices, zero_pose_local_translation=self.sk_local_translation)


if __name__ == '__main__':
    import joblib
    import os
    import pickle
    from poselib.poselib.visualization.common import plot_skeleton_H
    motion_path = 'test_data/converted_data/walking1-10_16.pkl'
    with open(motion_path,'rb') as f:
        data = joblib.load(f)

    file_name = os.path.basename(motion_path).split('.')[0]
    data = data[file_name]

    motion_global_rotation = torch.Tensor(data['pose_quat_global'])
    motion_root_translation = torch.Tensor(data['root_trans_offset'])

    with open('asset/smpl/smpl_skeleton_tree.pkl', 'rb') as f:
        skeleton_tree:SkeletonTree = pickle.load(f)

    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                     motion_global_rotation,
                                                     motion_root_translation,
                                                     is_local=False)
    sk_motion = SkeletonMotion.from_skeleton_state(sk_state,30)

    base_forward_model = BaseForwardModel(skeleton_tree)
    test_motion_global_rotation, test_motion_global_translation = \
        base_forward_model.forward_kinematics(motion_local_rotation=sk_motion.local_rotation,motion_root_translation=sk_motion.root_translation)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                    test_motion_global_rotation,
                                                                    motion_root_translation,
                                                                    is_local=False)
    new_sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, 30)

    print(f'max global rotation error: {(new_sk_motion.global_rotation-sk_motion.global_rotation).max()}')
    print(f'max translation error: {(test_motion_global_translation-sk_motion.global_translation).abs().max()}')

    plot_skeleton_H([sk_motion,new_sk_motion])
