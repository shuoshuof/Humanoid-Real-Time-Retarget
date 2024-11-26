# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 14:46
@Auth ： shuoshuof
@File ：VTRDYN_FULL.py
@Project ：Humanoid-Real-Time-Retarget
"""

import networkx as nx
VTRDYN_JOINT_NAMES = [
    'Hips',
    'LeftUpperLeg',
    'LeftLowerLeg',
    'LeftFoot',
    'RightUpperLeg',
    'RightLowerLeg',
    'RightFoot',
    'Spine',
    'Spine1',
    'Spine2',
    'Spine3',
    'LeftShoulder',
    'LeftUpperArm',
    'LeftLowerArm',
    'LeftHand',
    'LeftIndexFinger',
    'LeftIndexFinger1',
    'LeftIndexFinger2',
    'LeftIndexFinger3',
    'LeftMiddleFinger',
    'LeftMiddleFinger1',
    'LeftMiddleFinger2',
    'LeftMiddleFinger3',
    'LeftPinkyFinger',
    'LeftPinkyFinger1',
    'LeftPinkyFinger2',
    'LeftPinkyFinger3',
    'LeftRingFinger',
    'LeftRingFinger1',
    'LeftRingFinger2',
    'LeftRingFinger3',
    'LeftThumbFinger',
    'LeftThumbFinger1',
    'LeftThumbFinger2',
    'Neck',
    'Head',
    'RightShoulder',
    'RightUpperArm',
    'RightLowerArm',
    'RightHand',
    'RightIndexFinger',
    'RightIndexFinger1',
    'RightIndexFinger2',
    'RightIndexFinger3',
    'RightMiddleFinger',
    'RightMiddleFinger1',
    'RightMiddleFinger2',
    'RightMiddleFinger3',
    'RightPinkyFinger',
    'RightPinkyFinger1',
    'RightPinkyFinger2',
    'RightPinkyFinger3',
    'RightRingFinger',
    'RightRingFinger1',
    'RightRingFinger2',
    'RightRingFinger3',
    'RightThumbFinger',
    'RightThumbFinger1',
    'RightThumbFinger2'
]

# VTRDYN_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),
#                       (0,5),(5,6),(6,7),(7,8),
#                       (0,9),(9,10),(10,11),(11,12),(12,13),(13,14),
#                       (12,19),(19,20),(20,21),(21,22),
#                       (12,15),(15,16),(16,17),(17,18)]
VTRDYN_CONNECTIONS = [(0,1),(1,2),(2,3),
                      (0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(9,10),(10,11),(11,12),
                      (10,13),(13,14),(14,15),(15,16),
                      (10,17),(17,18),(18,19),(19,20),]

vtrdyn_graph = nx.DiGraph()


for i, keypoint_name in enumerate(VTRDYN_CONNECTIONS):
    vtrdyn_graph.add_node(i, label=keypoint_name)

vtrdyn_graph.add_edges_from(VTRDYN_CONNECTIONS)


vtrdyn_parent_indices = [-1] + [connection[0] for connection in VTRDYN_CONNECTIONS]


VTRDYN_JOINT_NAMES_LITE = ['Hips',
                          'RightUpperLeg',
                          'RightLowerLeg',
                          'RightFoot',
                           'LeftUpperLeg',
                           'LeftLowerLeg',
                           'LeftFoot',
                           'Spine',
                           'Spine1',
                           'Neck',
                           'Head',
                           'RightShoulder',
                           'RightUpperArm',
                           'RightLowerArm',
                           'RightHand',
                           'LeftShoulder',
                           'LeftUpperArm',
                           'LeftLowerArm',
                           'LeftHand']

VTRDYN_CONNECTIONS_LITE = [(0,1),(1,2),(2,3),
                      (0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(9,10),
                      (8,11),(11,12),(12,13),(13,14),
                      (8,15),(15,16),(16,17),(17,18),]

vtrdyn_lite_graph = nx.DiGraph()

for i, keypoint_name in enumerate(VTRDYN_JOINT_NAMES_LITE):
    vtrdyn_lite_graph.add_node(i, label=keypoint_name)

vtrdyn_lite_graph.add_edges_from(VTRDYN_CONNECTIONS_LITE)