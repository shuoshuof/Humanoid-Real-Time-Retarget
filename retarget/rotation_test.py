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
from retarget.spatial_transform.transform3d import *

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


def cal_shoulderPR(v1, v0, parent_global_rotation):
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

    shoulder_pitch_quat = quat_from_angle_axis(torch.tensor([theta1-theta0]),axis[1])

    phi1 = radians_between_vecs(v1_proj,v1,n=torch.cross(v1_proj,axis[1]))
    phi0 = radians_between_vecs(v0_proj,v0,n=torch.cross(v0_proj,axis[1]))

    print(f"rotation: {theta1-theta0}, {phi1-phi0}")

    shoulder_roll_quat = quat_from_angle_axis(phi1-phi0,axis[0])

    return shoulder_pitch_quat,shoulder_roll_quat

def cal_elbowP_and_shoulderY(v1, v0, parent_global_rotation):
    axis = torch.eye(3,dtype=torch.float32)
    parent_quat_inv = quat_inverse(parent_global_rotation)

    v1 = quat_rotate(parent_quat_inv,v1).squeeze(0)

    # v1 proj in xoy plane
    v1_proj = proj_in_plane(v1,axis[2])
    v0_proj = proj_in_plane(v0,axis[2])

    # shoulder yaw
    theta1 = radians_between_vecs(axis[0],v1_proj,n=axis[2])
    theta0 = radians_between_vecs(axis[0],v0_proj,n=axis[2])

    shoulder_yaw_quat = quat_from_angle_axis(torch.tensor([theta1-theta0]),axis[2])

    # elbow pitch
    phi1 = radians_between_vecs(v1_proj,v1,n=torch.cross(axis[2],v1_proj))
    phi0 = radians_between_vecs(v0_proj,v0,n=torch.cross(axis[2],v0_proj))


    print(f"rotation: {theta1 - theta0}, {phi1 - phi0}")

    elbow_pitch_quat = quat_from_angle_axis(phi1-phi0,axis[1])

    return shoulder_yaw_quat,elbow_pitch_quat



if __name__ == '__main__':
    x0,y0,z0 = 0.,0.,0.

    x1,y1,z1 = 0.,-1.,0.

    x2,y2,z2 = 0.,-1.,-1.

    x3,y3,z3 = 1.,-1.,-1.

    p0 = torch.tensor([x0,y0,z0]).unsqueeze(0)
    p1 = torch.tensor([x1,y1,z1]).unsqueeze(0)
    p2 = torch.tensor([x2,y2,z2]).unsqueeze(0)
    p3 = torch.tensor([x3,y3,z3]).unsqueeze(0)

    axis = torch.eye(3)


    vector0 = p1-p0
    vector1 = p2-p1
    vector2 = p3-p2

    quat0 = quat_from_angle_axis(torch.tensor([torch.pi/2*0], dtype=torch.float32), torch.tensor([0, 0, 1], dtype=torch.float32))
    # shoulder pitch
    quat1_1 = quat_from_angle_axis(torch.tensor([torch.pi/3*0], dtype=torch.float32), torch.tensor([0, 1, 0], dtype=torch.float32))
    # shoulder roll
    quat1_2 = quat_from_angle_axis(torch.tensor([-torch.pi/6*0], dtype=torch.float32),torch.tensor([1, 0, 0], dtype=torch.float32))
    # shoulder yaw
    quat1_3 = quat_from_angle_axis(torch.tensor([-torch.pi/6], dtype=torch.float32), torch.tensor([0, 0, 1], dtype=torch.float32))

    # elbow pitch
    quat2 = quat_from_angle_axis(torch.tensor([torch.pi/4], dtype=torch.float32), torch.tensor([0, 1, 0], dtype=torch.float32))

    quat1 = quat_mul_three(quat1_1,quat1_2,quat1_3)
                                                                                                                                                    
    axis1 = quat_rotate(quat0, axis)

    vector0_t = quat_rotate(quat0, vector0)
    vector1_t = quat_rotate(quat_mul_norm(quat0,quat1), vector1)

    vector2_t = quat_rotate(quat_mul_norm(quat_mul_norm(quat0,quat1), quat2), vector2)


    pitch_joint_quat, roll_joint_quat = cal_shoulderPR(v1=vector1_t[0], v0=vector1[0], parent_global_rotation=quat0)


    combine_quat = quat_mul_three(quat0,pitch_joint_quat,roll_joint_quat)
    vector1_t_cal = quat_rotate(combine_quat, vector1)


    assert torch.allclose(vector1_t_cal,vector1_t,rtol=1e-3,atol=1e-6)

    shoulder_global_rotation = combine_quat
    shoulder_yaw_quat,elbow_pitch_quat = cal_elbowP_and_shoulderY(v1=vector2_t[0], v0=vector2[0], parent_global_rotation=combine_quat)

    # TODO
    vector2_t_cal = quat_rotate(quat_mul_three(combine_quat,shoulder_yaw_quat,elbow_pitch_quat), vector2)

    assert torch.allclose(vector2_t_cal,vector2_t,rtol=1e-3,atol=1e-6)

    plt = Plotter()
    plt.add(Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis),
            Axes(center_pos=torch.tensor([x0, y0, z0]), axes=axis1),
            Ray(p0,p1),
            Ray(p1,p2),
            Ray(p2,p3),
            Ray(torch.zeros((1,3)),vector0_t),
            Ray(vector0_t,vector0_t+vector1_t),
            Ray(vector0_t+vector1_t,vector0_t+vector1_t+vector2_t),
            )
    plt.show(axes=1,viewup='z')

















