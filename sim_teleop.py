# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/20 11:54
@Auth ： shuoshuof
@File ：sim_teleop.py
@Project ：Humanoid-Real-Time-Retarget
"""
import threading
import pandas as pd
from collections import OrderedDict
import time
import pickle
import cv2

from robot_kinematics_model import RobotZeroPose

from retarget.utils import get_mocap_translation
from retarget.spatial_transform.transform3d import *
from retarget.torch_ext import to_torch, to_numpy

from retarget.retarget_solver import HuUpperBodyFromMocapRetarget
from mocap_communication.receive import MocapReceiver
from sim.mocap_env import MocapControlEnv


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

    df = pd.read_csv('test_motion/mocap_raw/walk_with_hand.csv')
    motion_global_translation = get_mocap_translation(df)
    motion_global_translation = to_torch(motion_global_translation)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'rb') as f:
        vtrdyn_zero_pose = pickle.load(f)

    vtrdyn_zero_pose = RobotZeroPose.from_skeleton_state(vtrdyn_zero_pose)
    hu_zero_pose = RobotZeroPose.from_urdf('asset/hu/hu_v5.urdf')

    hu_retarget = HuUpperBodyFromMocapRetarget(vtrdyn_zero_pose,hu_zero_pose)
    mocap_control_env = MocapControlEnv()

    recorder = DataRecorder('data')
    receiver = MocapReceiver('192.168.1.167', 12345)

    receiver_thread = threading.Thread(target=receiver.run)
    receiver_thread.start()
    if receiver.has_connected.wait(timeout=20):
    # if True:
        last_dof_pose = torch.zeros(31 - 1)
        while True:
            start = time.time()
            if not receiver.is_connected():
                break

            body_pose = receiver.get_body_pose()

            if body_pose is not None:
                body_pose = to_torch(body_pose)
                body_pose = body_pose[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
                _, dof_pos = hu_retarget.retarget_from_global_translation(body_pose)
                last_dof_pose = dof_pos
            else:
                dof_pos = last_dof_pose
            # _, dof_pos = hu_retarget.retarget_from_global_translation(motion_global_translation[i])
            dof_state, viewer_img = mocap_control_env.step(dof_pos)
            recorder.record(body_pose,dof_pos,dof_state,viewer_img)

            end = time.time()
            # print(f"{end - start:.3f}")

    receiver.stop()
    receiver_thread.join()

    recorder.save()


