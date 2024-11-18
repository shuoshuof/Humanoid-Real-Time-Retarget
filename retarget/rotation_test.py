# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 17:17
@Auth ： shuoshuof
@File ：test.py
@Project ：Humanoid-Real-Time-Retarget
"""

from vedo import *
import numpy as np
import torch

from retarget.torch_ext import to_numpy

from poselib.poselib.core.rotation3d import *


class Axes(Arrows):
    def __init__(self, center_pos, axes):
        axes = to_numpy(axes)
        start  = to_numpy(center_pos.unsqueeze(0).repeat(3,1))
        end = start + axes
        super().__init__(start, end, c='blue',alpha=0.5,thickness=0.1)
class Ray(Lines):
    def __init__(self,start,end):
        start = to_numpy(start)
        end = to_numpy(end)
        super().__init__(start,end,c='red',lw=3)

def proj_in_plane(v, n):
    r"""
    project a vector in a plane
    :param v: vector
    :param n:  the normal of the plane
    :return:
    """
    n_norm = torch.linalg.norm(n)
    assert n_norm>1e-6

    v_proj_n = (torch.dot(v,n)/n_norm**2)*n
    v_proj = v - v_proj_n

    return v_proj,v_proj_n

def radians_between_vecs(v1,v2):
    pass

def cal_spherical_angle(v1,v0,parent_global_rotation,plane='xoz'):
    parent_orig_axis = torch.eye(3)
    parent_axis = quat_rotate(parent_global_rotation,torch.eye(3))
    if plane=='xoz':
        n_xoz = parent_axis[[1]]
        v1_proj,v1_proj_n = proj_in_plane(v1, n_xoz)
        angle = torch.arccos(torch.dot(v1_proj,parent_orig_axis[0])/(torch.linalg.norm(v1_proj_n)*1))
        cross_product = torch.cross(v1_proj, parent_orig_axis[0])
        if torch.dot(cross_product, n_xoz) < 0:
            angle = -angle

        v0_proj,v0_proj_n = proj_in_plane(v0,parent_orig_axis[[1]])




        return angle

if __name__ == '__main__':
    x0,y0,z0 = 0,0,0

    x1,y1,z1 = 0,-1,0

    x2,y2,z2 = 0,-1,-1

    axis = torch.eye(3)


    vector0 = torch.tensor([x1,y1,z1]).unsqueeze(0)
    vector1 = torch.tensor([x2,y2,z2])-vector0

    quat0 = quat_from_angle_axis(torch.tensor([torch.pi / 2], dtype=torch.float32), torch.tensor([0, 0, 1], dtype=torch.float32))

    quat1 = quat_from_angle_axis(torch.tensor([torch.pi / 2], dtype=torch.float32), torch.tensor([1, 1, 0], dtype=torch.float32))

    axis1 = quat_rotate(quat0, axis)

    vector0_t = quat_rotate(quat0, vector0)
    vector1_t = quat_rotate(quat0,quat_rotate(quat1, vector1))

    print(axis1)



    plt = Plotter()
    plt.add(Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis),
            Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis1),
            Ray(torch.zeros((1,3)),vector0),
            Ray(vector0,vector0+vector1),
            Ray(torch.zeros((1,3)),vector0_t),
            Ray(vector0_t,vector0_t+vector1_t),
            )
    plt.show(axes=1,viewup='z')
















