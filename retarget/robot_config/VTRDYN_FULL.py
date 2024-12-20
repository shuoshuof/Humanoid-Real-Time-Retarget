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
    'LeftIndexFinger1', #2
    'LeftIndexFinger2',
    'LeftIndexFinger3',
    'LeftMiddleFinger',
    'LeftMiddleFinger1',#6
    'LeftMiddleFinger2',
    'LeftMiddleFinger3',
    'LeftPinkyFinger',
    'LeftPinkyFinger1',#10
    'LeftPinkyFinger2',
    'LeftPinkyFinger3',
    'LeftRingFinger',
    'LeftRingFinger1',#14
    'LeftRingFinger2',
    'LeftRingFinger3',
    'LeftThumbFinger',#17
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

# VTRDYN_JOINT_NAMES = [
#     'Hips',
#     'LeftUpperLeg',
#     'LeftLowerLeg',
#     'LeftFoot',
#     # 'LeftToe',
#     'RightUpperLeg',
#     'RightLowerLeg',
#     'RightFoot',
#     # 'RightToe',
#     'Spine',
#     'Spine1',
#     'Spine2',
#     'Spine3',
#     'Neck',
#     'Head',
#
#     'LeftShoulder',
#     'LeftUpperArm',
#     'LeftLowerArm',
#     'LeftHand',
#
#     'RightShoulder',
#     'RightUpperArm',
#     'RightLowerArm',
#     'RightHand',
#
#     'LeftThumbFinger',
#     'LeftThumbFinger1',
#     'LeftThumbFinger2',
#     'LeftIndexFinger',
#     'LeftIndexFinger1',
#     'LeftIndexFinger2',
#     'LeftIndexFinger3',
#     'LeftMiddleFinger',
#     'LeftMiddleFinger1',
#     'LeftMiddleFinger2',
#     'LeftMiddleFinger3',
#     'LeftRingFinger',
#     'LeftRingFinger1',
#     'LeftRingFinger2',
#     'LeftRingFinger3',
#     'LeftPinkyFinger',
#     'LeftPinkyFinger1',
#     'LeftPinkyFinger2',
#     'LeftPinkyFinger3',
#
#     'RightThumbFinger',
#     'RightThumbFinger1',
#     'RightThumbFinger2',
#     'RightIndexFinger',
#     'RightIndexFinger1',
#     'RightIndexFinger2',
#     'RightIndexFinger3',
#     'RightMiddleFinger',
#     'RightMiddleFinger1',
#     'RightMiddleFinger2',
#     'RightMiddleFinger3',
#     'RightRingFinger',
#     'RightRingFinger1',
#     'RightRingFinger2',
#     'RightRingFinger3',
#     'RightPinkyFinger',
#     'RightPinkyFinger1',
#     'RightPinkyFinger2',
#     'RightPinkyFinger3',
# ]
# receive order
BodyNodes = [
    'Hips',
    'RightUpperLeg',
    'RightLowerLeg',
    'RightFoot',
    'RightToe',
    'LeftUpperLeg',
    'LeftLowerLeg',
    'LeftFoot',
    'LeftToe',
    'Spine',
    'Spine1',
    'Spine2',
    'Spine3',
    'Neck',
    'Head',
    'RightShoulder',
    'RightUpperArm',
    'RightLowerArm',
    'RightHand',
    'LeftShoulder',
    'LeftUpperArm',
    'LeftLowerArm',
    'LeftHand',
]

HandNodes_20_r = [
    'RightHand',
    'RightThumbFinger',
    'RightThumbFinger1',
    'RightThumbFinger2',
    'RightIndexFinger',
    'RightIndexFinger1',
    'RightIndexFinger2',
    'RightIndexFinger3',
    'RightMiddleFinger',
    'RightMiddleFinger1',
    'RightMiddleFinger2',
    'RightMiddleFinger3',
    'RightRingFinger',
    'RightRingFinger1',
    'RightRingFinger2',
    'RightRingFinger3',
    'RightPinkyFinger',
    'RightPinkyFinger1',
    'RightPinkyFinger2',
    'RightPinkyFinger3',
]

HandNodes_20_l = [
    'LeftHand',
    'LeftThumbFinger',
    'LeftThumbFinger1',
    'LeftThumbFinger2',
    'LeftIndexFinger',
    'LeftIndexFinger1',
    'LeftIndexFinger2',
    'LeftIndexFinger3',
    'LeftMiddleFinger',
    'LeftMiddleFinger1',
    'LeftMiddleFinger2',
    'LeftMiddleFinger3',
    'LeftRingFinger',
    'LeftRingFinger1',
    'LeftRingFinger2',
    'LeftRingFinger3',
    'LeftPinkyFinger',
    'LeftPinkyFinger1',
    'LeftPinkyFinger2',
    'LeftPinkyFinger3',
]