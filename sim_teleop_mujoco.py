# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/1 18:32
@Auth ： shuoshuof
@File ：sim_teleop_mujoco.py
@Project ：Humanoid-Real-Time-Retarget
"""

import threading
import pandas as pd
from collections import OrderedDict
import time
import pickle
import cv2
from sim.mujoco_teleop_env import MujocoTeleopEnv
from robot_kinematics_model.base_robot import RobotZeroPose
from poselib.poselib.core.rotation3d import *
from retarget.utils.parse_mocap import get_vtrdyn_translation,vtrdyn_zero_pose_transform,vtrdyn_broadcast_zero_pose_transform
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch, to_numpy

from retarget.retarget_solver import VtrdynFullBodyPosRetargeter
from mocap_communication.mocap_receiver import MocapReceiver



class DataRecorder:
    def __init__(self,save_dir):
        self.dof_pos = []
        self.dof_state = []
        self.img = []
        self.save_dir = save_dir

    def record(self,dof_pos,dof_state,img):
        # map_indices = [14,15,16,17,18, 23,24,25,26,27]
        map_indices = np.arange(len(dof_pos))
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
    teleop_env = MujocoTeleopEnv('asset/hu/scene.xml')
    teleop_env.launch_viewer()

    recorder = DataRecorder('data')
    receiver = MocapReceiver('192.168.1.174', 12345)

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
            # dof_state, viewer_img = mocap_control_env.step(dof_pos)
            teleop_env.step(dof_pos)
            # recorder.record(dof_pos, dof_state, viewer_img)
            img = teleop_env.get_camera_image()
            img = cv2.flip(img, 0)  # Flip the image vertically
            cv2.imshow("MuJoCo Camera Image", img[..., ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end = time.time()
            # print(f"{end - start:.3f}")

    receiver.stop()
    receiver_thread.join()
    # with open('data/test_data2.pkl','wb') as f:
    #     pickle.dump(data_list,f)
    # recorder.save()