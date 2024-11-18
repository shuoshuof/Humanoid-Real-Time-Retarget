# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 11:44
@Auth ： shuoshuof
@File ：main.py
@Project ：Humanoid-Real-Time-Retarget
"""
import pandas as pd
import numpy as np

from retarget.utils import get_mocap_translation,parse_urdf
from spatial_transform.transform3d import *















df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
motion_global_translation = get_mocap_translation(df)







