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

    return v_proj

def radians_between_vecs(v1,v2,n):
    r"""
    calculate the angle between two vectors
    :param v1:
    :param v2:
    :param n:
    :return:
    """

    v1 = v1 / torch.linalg.norm(v1)
    v2 = v2 / torch.linalg.norm(v2)
    normal = n / torch.linalg.norm(n)

    cos_theta = torch.dot(v1, v2).clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)

    cross = torch.cross(v1, v2)
    direction = torch.dot(normal, cross)

    if direction < 0:
        angle = -angle

    return angle

# def cal_shoulder_spherical_joint_rotation(v1,v0,parent_global_rotation,plane='xoz'):
#     r"""
#     calculate shoulder spherical joint rotation
#     the order is pitch and roll. the raw need to use the child joint to cal
#     :param v1:
#     :param v0:
#     :param parent_global_rotation:
#     :param plane:
#     :return:
#     """
#     parent_orig_axis = torch.eye(3,dtype=torch.float32)
#     # axis in parent frame
#     parent_axis = quat_rotate(parent_global_rotation,torch.eye(3,dtype=torch.float32))
#     if plane=='xoz':
#         n_xoz = parent_axis[1]
#         v1_proj = proj_in_plane(v1, n_xoz)
#         v0_proj = proj_in_plane(v0, parent_orig_axis[1])
#
#         theta1 = radians_between_vecs(parent_axis[0],v1_proj,n=parent_axis[1])
#         theta0 = radians_between_vecs(parent_orig_axis[0],v0_proj,n=parent_orig_axis[1])
#
#         # quat pitch (rotation in xoz plane)
#         pitch_joint_quat = quat_from_angle_axis(torch.tensor([theta1-theta0]),torch.tensor([0.,1.,0.]))
#
#         pitch_joint_axis = quat_rotate(pitch_joint_quat,parent_axis)
#
#         phi1 = radians_between_vecs(pitch_joint_axis[2],v1,n=pitch_joint_axis[0])
#         phi0 = radians_between_vecs(parent_orig_axis[2],v0,n=parent_orig_axis[0])
#
#         roll_joint_quat = quat_from_angle_axis(phi1-phi0,torch.tensor([1.,0.,0.]))
#
#         return pitch_joint_quat,roll_joint_quat
#
#     else:
#         raise NotImplementedError

def cal_shoulder_spherical_joint_rotation(v1,v0,parent_global_rotation):
    r"""
    calculate shoulder spherical joint rotation
    the order is pitch and roll. the raw need to use the child joint to cal
    :param v1:
    :param v0:
    :param parent_global_rotation:
    :param plane:
    :return:
    """
    axis = torch.eye(3,dtype=torch.float32)

    parent_quat_inv = quat_inverse(parent_global_rotation)

    v1 = quat_rotate(parent_quat_inv,v1).squeeze(0)

    # v1 proj in xoz plane
    v1_proj = proj_in_plane(v1,axis[1])

    v0_proj = proj_in_plane(v0,axis[1])

    theta1 = radians_between_vecs(axis[0],v1_proj,n=axis[1])
    theta0 = radians_between_vecs(axis[0],v0_proj,n=axis[1])

    pitch_joint_quat = quat_from_angle_axis(torch.tensor([theta1-theta0]),axis[1])

    phi1 = radians_between_vecs(v1_proj,v1,n=torch.cross(v1_proj,axis[1]))
    phi0 = radians_between_vecs(v0_proj,v0,n=torch.cross(v0_proj,axis[1]))

    print(f"rotation: {theta1-theta0}, {phi1-phi0}")

    roll_joint_quat = quat_from_angle_axis(phi1-phi0,axis[0])

    return pitch_joint_quat,roll_joint_quat




if __name__ == '__main__':
    x0,y0,z0 = 0.,0.,0.

    x1,y1,z1 = 0.,-1.,0.

    x2,y2,z2 = 0.,-1.,-1.

    axis = torch.eye(3)


    vector0 = torch.tensor([x1,y1,z1]).unsqueeze(0)
    vector1 = torch.tensor([x2,y2,z2])-vector0

    quat0 = quat_from_angle_axis(torch.tensor([torch.pi/3], dtype=torch.float32), torch.tensor([0, 0, 1], dtype=torch.float32))

    quat1_1 = quat_from_angle_axis(torch.tensor([torch.pi / 5], dtype=torch.float32), torch.tensor([1, 0, 0], dtype=torch.float32))
    quat1_2 = quat_from_angle_axis(torch.tensor([torch.pi / 6], dtype=torch.float32), torch.tensor([0, 1, 0], dtype=torch.float32))

    quat1 = quat_mul_norm(quat1_2,quat1_1)
                                                                                                                                                    
    axis1 = quat_rotate(quat0, axis)

    vector0_t = quat_rotate(quat0, vector0)
    vector1_t = quat_rotate(quat_mul_norm(quat0,quat1), vector1)

    pitch_joint_quat, roll_joint_quat = cal_shoulder_spherical_joint_rotation(v1=vector1_t[0], v0=vector1[0],parent_global_rotation=quat0)


    combine_quat = quat_mul_three(quat0,pitch_joint_quat,roll_joint_quat)
    # vector1_t_cal = quat_rotate(pitch_joint_quat, quat_rotate(roll_joint_quat, quat_rotate(quat0, vector1)))
    vector1_t_cal = quat_rotate(combine_quat, vector1)


    assert vector1_t_cal.allclose(vector1_t)

    plt = Plotter()
    plt.add(Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis),
            Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis1),
            Ray(torch.zeros((1,3)),vector0),
            Ray(vector0,vector0+vector1),
            Ray(torch.zeros((1,3)),vector0_t),
            Ray(vector0_t,vector0_t+vector1_t),
            )
    plt.show(axes=1,viewup='z')

















