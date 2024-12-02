# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/2 16:17
@Auth ： shuoshuof
@File ：dof_cfg.py
@Project ：Humanoid-Real-Time-Retarget
"""
from collections import OrderedDict

Hu_DOF_CTRL = OrderedDict(
    kp = [
        500, 300, 100, 200, 50,
        500, 300, 100, 200, 50,
        300,
        600, 200, 200, 200, 60, 60, 60, 100, 100,
        600, 200, 200, 200, 60, 60, 60, 100, 100,
        100.*3/5,
    ],
    kd = [
        5, 5, 5, 6, 1,
        5, 5, 5, 6, 1,
        6,
        20, 20, 7, 7, 1.8, 1.8, 1.8, 1, 1,
        20, 20, 7, 7, 1.8, 1.8, 1.8, 1, 1,
        1,
    ]
)
