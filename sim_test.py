# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 10:52
@Auth ： shuoshuof
@File ：sim_test.py
@Project ：Humanoid-Real-Time-Retarget
"""
import numpy as np
import time
import cv2

from mocap_communication.server_receive import Receiver
from mocap_communication.server_send import Transmitter

from sim.isaac_teleop_env import MocapControlEnv
from retarget.torch_ext import to_torch
def process_data(img,dof_state):
    img = img[:, 180:-180]
    img = cv2.resize(img.astype(np.uint8), (224, 224))
    img = img[np.newaxis,...]
    assert img.shape == (1,224,224,3)

    dof_state = dof_state['pos'].astype(np.float32)[np.newaxis,...]

    return {"img": img,"dof_state": dof_state}


if __name__ == '__main__':

    trans = Transmitter('192.168.1.13', 6000)
    trans.connect()
    recv = Receiver('192.168.1.157', 37535)#35600 37535
    # recv = Receiver('192.168.1.13', 7000)
    recv.connect()

    env = MocapControlEnv()
    dof_state,viewer_img =  env.step(dof_tar_pos=None)
    data_dict = process_data(viewer_img,dof_state)

    trans.send(data_dict)

    while True:
        received_dict = recv.receive()
        if received_dict is not None:
            print(received_dict)
            print("Received:", received_dict['dof_pos'].shape)
            time.sleep(0.001)
            for i,dof_pos in enumerate(received_dict['dof_pos']):
                env.step(dof_tar_pos=dof_pos)
                dof_state, viewer_img = env.step(None)
            data_dict = process_data(viewer_img, dof_state)
            trans.send(data_dict)
