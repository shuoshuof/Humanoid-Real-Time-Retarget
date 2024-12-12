# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/11 17:34
@Auth ： shuoshuof
@File ：get_hu_pose.py
@Project ：data_convert
"""
import numpy as np
from urdfpy import URDF
from collections import OrderedDict
import networkx as nx
import copy
import pickle

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState

def _parse_urdf(urdf_path):
    urdf_robot: URDF = URDF.load(urdf_path)
    # urdf_robot.show()
    fk_link = urdf_robot.link_fk()

    urdf_graph = copy.deepcopy(urdf_robot._G)

    link_parents = np.argmax(nx.adjacency_matrix(urdf_graph).todense(), axis=1).A1
    link_parents[0] = -1

    link_names = []
    link_mesh_file_names = []
    link_translations = []

    for link, transform in fk_link.items():
        link_name = link.name
        link_translation = transform[:3, 3]

        link_mesh_file_names.append(link.visuals[0].geometry.mesh.filename)
        link_names.append(link_name)
        link_translations.append(link_translation)
    link_translations = np.array(link_translations)
    link_local_translations = np.zeros_like(link_translations)
    link_local_translations[1:] = link_translations[1:] - link_translations[link_parents[1:]]

    print(link_names)
    print(link_parents)

    robot_sk_tree = SkeletonTree.from_dict(
        OrderedDict({'node_names': link_names,
                     'parent_indices': {'arr': np.array(link_parents), 'context': {'dtype': 'int64'}},
                     'local_translation': {'arr': np.array(link_local_translations), 'context': {'dtype': 'float32'}}})
    )

    robot_zero_pose = SkeletonState.zero_pose(robot_sk_tree)
    from poselib.poselib.visualization.common import plot_skeleton_H
    plot_skeleton_H([robot_zero_pose])
    return robot_zero_pose, link_mesh_file_names

if __name__ == "__main__":
    zero_pose, mesh_files = _parse_urdf("asset/hu/hu_v5.urdf")
    with open("asset/hu_pose/hu_v5_zero_pose.pkl", "wb") as f:
        pickle.dump(zero_pose, f)



