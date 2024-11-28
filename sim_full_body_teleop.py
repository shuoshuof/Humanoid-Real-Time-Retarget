# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/27 17:17
@Auth ： shuoshuof
@File ：sim_full_body_teleop.py
@Project ：Humanoid-Real-Time-Retarget
"""

import threading
import pandas as pd
from collections import OrderedDict
import time
import pickle
import cv2
from sim.mocap_env import MocapControlEnv
from robot_kinematics_model.base_robot import RobotZeroPose
from poselib.poselib.core.rotation3d import *
from retarget.utils.parse_mocap import get_vtrdyn_translation,vtrdyn_zero_pose_transform,vtrdyn_broadcast_zero_pose_transform
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch, to_numpy

from retarget.retarget_solver import VtrdynFullBodyPosRetargeter
from mocap_communication.mocap_receiver import MocapReceiver



class DataRecorder:
    def __init__(self,save_dir):
        self.body_pose = []
        self.dof_pos = []
        self.dof_state = []
        self.img = []
        self.save_dir = save_dir

    def record(self,body_pose,dof_pos,dof_state,img):
        # map_indices = [14,15,16,17,18, 23,24,25,26,27]
        map_indices = np.arange(len(dof_pos))
        if body_pose is not None:
            self.body_pose.append(to_numpy(body_pose))
        self.dof_pos.append(to_numpy(dof_pos).astype(np.float32)[map_indices])
        self.dof_state.append(self.process_dof_state(dof_state)[map_indices])
        self.img.append(self.process_img(img))

    def process_img(self,img):
        assert img.shape == (720,1280,3)
        img = img[:,180:-180]
        img = cv2.resize(img.astype(np.uint8),(224,224))
        return img
    def process_dof_state(self,dof_state):
        dof_state = dof_state['pos']
        return dof_state.astype(np.float32)

    def save(self):
        data_dict = OrderedDict(
            body_pos=np.stack(self.body_pose),
            dof_pos=np.stack(self.dof_pos),
            dof_state=np.stack(self.dof_state),
            img=np.stack(self.img)
        )

        save_time = time.strftime("%Y-%m-%d_%H:%M:%S")

        with open(f'{self.save_dir}/{save_time}.pkl','wb') as f:
            pickle.dump(data_dict,f)

        print('save data successfully')


if __name__ == '__main__':
    with open('asset/zero_pose/vtrdyn_full_zero_pose.pkl', 'rb') as f:
        vtrdyn_full_zero_pose = pickle.load(f)

    vtrdyn_full_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_full_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    hu_retarget = VtrdynFullBodyPosRetargeter(vtrdyn_full_zero_pose,hu_zero_pose)
    mocap_control_env = MocapControlEnv()

    recorder = DataRecorder('data')
    receiver = MocapReceiver('192.168.1.167', 12345)

    receiver_thread = threading.Thread(target=receiver.run)
    receiver_thread.start()

    data_list = []

    if receiver.has_connected.wait(timeout=20):
    # if True:
        last_dof_pose = torch.zeros(31 - 1)
        while True:
            start = time.time()
            if not receiver.is_connected():
                break

            data_dict = receiver.get_data_dict()
            body_pos = data_dict['body_pos']

            if data_dict is not None and not np.allclose(body_pos,0):
                data_list.append(data_dict)
                # preprocess data
                # body_quat = to_torch(data_dict['body_quat'])
                # body_quat = body_quat[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
                # full_body_quat = quat_identity([59,])
                # full_body_quat[[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]] = body_quat
                # full_body_quat = vtrdyn_full_zero_pose_transform(full_body_quat)
                # body_quat = full_body_quat[[[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 34, 35, 36, 37, 38, 39, 11, 12, 13, 14]]]

                # body_quat = to_torch(data_dict['body_quat'])
                # body_quat = body_quat[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
                # body_quat = vtrdyn_broadcast_zero_pose_transform(body_quat)



                body_pos = to_torch(data_dict['body_pos'])
                body_pos = body_pos[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]

                left_hand_pos = to_torch(data_dict['left_hand_pos'])[[0, 4,5,6,7, 8,9,10,11, 16,17,18,19, 12,13,14,15, 1,2,3]]
                right_hand_pos = to_torch(data_dict['right_hand_pos'])[[0, 4,5,6,7, 8,9,10,11, 16,17,18,19, 12,13,14,15, 1,2,3]]


                _, dof_pos,_ = hu_retarget.retarget(
                    body_pos,
                    left_hand_pos,
                    right_hand_pos
                )

                last_dof_pose = dof_pos
            else:
                dof_pos = last_dof_pose
            # _, dof_pos = hu_retarget.retarget_from_global_translation(motion_global_translation[i])
            dof_state, viewer_img = mocap_control_env.step(dof_pos)
            recorder.record(body_pos, dof_pos, dof_state, viewer_img)

            end = time.time()
            # print(f"{end - start:.3f}")

    receiver.stop()
    receiver_thread.join()
    with open('data/test_data2.pkl','wb') as f:
        pickle.dump(data_list,f)
    # recorder.save()
