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
    quat_x = sRot.from_euler('x', euler_angles[...,0], degrees=False).as_quat()
    quat_y = sRot.from_euler('y', euler_angles[...,1], degrees=False).as_quat()
    quat_z = sRot.from_euler('z', euler_angles[...,2], degrees=False).as_quat()

    return torch.Tensor(quat_x), torch.Tensor(quat_y), torch.Tensor(quat_z)



def cal_spherical_angle(vector,vector0,parent_global_rotation):
    pass


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



