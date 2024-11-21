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
    def __init__(self,pos,r=0.02):
        super().__init__(pos,r=r)
