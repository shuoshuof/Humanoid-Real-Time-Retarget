import numpy as np
import pickle
import copy
from collections import OrderedDict

import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H

from body_visualizer.visualizer import BodyVisualizer

from motion_convert.robot_config.Hu import hu_graph,HU_JOINT_NAMES
from motion_convert.utils.transform3d import coord_transform

if __name__ == '__main__':
    with open('asset/zero_pose/hu_zero_pose.pkl','rb') as f:
        hu_zero_pose:SkeletonState = pickle.load(f)
    hu_sk_tree = hu_zero_pose.skeleton_tree
    hu_local_translation = hu_sk_tree.local_translation.clone()

    new_hu_joint_names = copy.deepcopy(HU_JOINT_NAMES)
    # new_hu_joint_names+= ['left_end_link', ]

    hu_local_translation[[6]] = torch.Tensor([[1.0,0.0,0.0]])
    hu_local_translation[[12]] = torch.Tensor([[1.0,0.0,0.0]])


    new_hu_sk_tree = SkeletonTree.from_dict(
        OrderedDict({'node_names': hu_sk_tree.node_names,
                    'parent_indices': {'arr': hu_sk_tree.parent_indices.numpy(), 'context': {'dtype': 'int64'}},
                    'local_translation':{'arr': hu_local_translation.numpy(), 'context': {'dtype': 'float32'}}})
    )

    new_hu_zero_pose = SkeletonState.zero_pose(new_hu_sk_tree)

    with open('asset/zero_pose/new_hu_zero_pose.pkl','wb') as f:
        pickle.dump(new_hu_zero_pose,f)

    bd_vis = BodyVisualizer(hu_graph)
    pos = new_hu_zero_pose.global_translation

    bd_vis.step(pos)

    plot_skeleton_H([new_hu_zero_pose])