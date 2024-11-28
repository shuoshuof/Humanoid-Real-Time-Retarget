# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/21 15:42
@Auth ： shuoshuof
@File ：vedo_geoms.py
@Project ：Humanoid-Real-Time-Retarget
"""
from vedo import *

class SkeletonLines(Lines):
    def __init__(self,start,end):
        super().__init__(start,end,lw=3,c='blue')

class SkeletonJoints(Spheres):
    def __init__(self,pos,r=0.02,colors=None):
        super().__init__(pos,r=r,c=colors,alpha=0.5)

class JointAxes(Arrows):
    def __init__(self,start,end,color):
        super().__init__(start,end,c=color)
