import numpy as np
import torch
from poselib.poselib.core.rotation3d import *
from typing import Dict
import copy
from scipy.spatial.transform import Rotation as sRot

@torch.jit.script
def quat_between_two_vecs(vec1, vec2):
    '''calculate a quaternion that rotates from vector v1 to vector v2'''
    if torch.norm(vec1,dim=-1).max() <= 1e-6 or torch.norm(vec2,dim=-1).max() <= 1e-6:
        return torch.tensor([[0, 0, 0, 1]]*vec1.shape[0], dtype=torch.float32)

    vec1 = vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)
    vec2 = vec2 / torch.linalg.norm(vec2, dim=-1, keepdim=True)
    cross_prod = torch.cross(vec1, vec2, dim=-1)
    dots = torch.sum(vec1 * vec2, dim=-1, keepdim=True)
    real_part = (1 + dots)  # Adding 1 to ensure the angle calculation is stable
    quat = torch.cat([cross_prod, real_part], dim=-1)
    quat = quat_normalize(quat)
    return quat


def coord_transform(p,order:list=None,dir=None):
    if order is not None:
        p = p[...,order]
    if dir is not None:
        p = p* dir
    return p

@torch.jit.script
def cal_joint_quat(zero_pose_local_translation, motion_local_translation):
    r"""
    refer to https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    :param zero_pose_local_translation:
    :param motion_local_translation:
    :return: quat
    """
    A = torch.einsum('bij,bjk->bik', motion_local_translation.permute(0, 2, 1), zero_pose_local_translation)
    U, _, Vt = torch.linalg.svd(A)
    R_matrix = torch.einsum('bij,bjk->bik', U, Vt)

    det = torch.linalg.det(R_matrix)
    Vt[det < 0, -1, :] *= -1
    R_matrix = torch.einsum('bij,bjk->bik', U, Vt)

    # rotation = sRot.from_matrix(R_matrix)
    # quats = rotation.as_quat()
    quats = quat_from_rotation_matrix(R_matrix)
    return quats

def quat_in_xyz_axis(q,seq:str='xyz'):
    r = sRot.from_quat(q)
    euler_angles = r.as_euler(seq, degrees=False)
    quat_1 = sRot.from_euler(seq[0], euler_angles[...,0], degrees=False).as_quat()
    quat_2 = sRot.from_euler(seq[1], euler_angles[...,1], degrees=False).as_quat()
    quat_3 = sRot.from_euler(seq[2], euler_angles[...,2], degrees=False).as_quat()

    return torch.Tensor(quat_1), torch.Tensor(quat_2), torch.Tensor(quat_3)

@torch.jit.script
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

@torch.jit.script
def radians_between_vecs(v1,v2,n):
    r"""
    calculate the angle between two vectors
    pay attention the direction of the normal
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

    angle = angle*torch.sign(direction)

    return angle

# # @torch.jit.script
# def proj_in_plane(v, n):
#     r"""
#     project a vector in a plane
#     :param v: vector
#     :param n:  the normal of the plane
#     :return:
#     """
#     n_norm = torch.linalg.norm(n,dim=-1)
#     assert torch.all(n_norm>1e-6,dim=-1)
#
#     # v_proj_n = (torch.dot(v,n.view(-1,3))/n_norm**2)*n
#     v_proj_n = (torch.sum(v*n.view(-1,3),dim=-1)/n_norm**2).unsqueeze(-1)*n.view(-1,3)
#     assert v_proj_n.shape == v.shape
#     v_proj = v - v_proj_n
#     return v_proj
#
# # @torch.jit.script
# def radians_between_vecs(v1,v2,n):
#     r"""
#     calculate the angle between two vectors
#     :param v1:
#     :param v2:
#     :param n:
#     :return:
#     """
#
#     v1 = v1 / torch.linalg.norm(v1,dim=-1)
#     v2 = v2 / torch.linalg.norm(v2,dim=-1).view(-1,1)
#     normal = n / torch.linalg.norm(n,dim=-1)
#
#     # cos_theta = torch.dot(v1, v2).clamp(-1.0, 1.0)
#     cos_theta = torch.sum(v1*v2,dim=-1).clamp(-1.0, 1.0)
#     angle = torch.acos(cos_theta)
#
#     cross = torch.cross(v1.view(-1,3), v2.view(-1,3))
#     # direction = torch.dot(normal, cross)
#     direction = torch.sum(normal * cross, dim=-1)
#
#     angle = angle*torch.sign(direction)
#
#     return angle


@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

@torch.jit.script
def quat_slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

@torch.jit.script
def quat_to_dof_pos(quat,dof_axis):
    # type: (Tensor, List[int]) -> Tensor
    axis = torch.eye(3)
    axis = axis[dof_axis]
    exp_map = quat_to_exp_map(quat)
    dof_pos = exp_map*axis
    return dof_pos[torch.arange(len(dof_axis)),dof_axis]


