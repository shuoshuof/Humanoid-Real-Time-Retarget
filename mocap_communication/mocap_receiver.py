# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/19 18:15
@Auth ： shuoshuof
@File ：recieve.py
@Project ：Humanoid-Real-Time-Retarget
"""

import socket
import pickle
import numpy as np
import time
import threading

class MocapReceiver:
    def __init__(self, host, port, buffer_size=8192):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.client_socket = None
        self.data_dict = {
            'body_pos': np.zeros((23, 3),dtype=np.float32),
            'body_quat': np.array([[0., 0., 0.,1.]] * 23,dtype=np.float32),
            'left_hand_pos': np.zeros((20,3),dtype=np.float32),
            'right_hand_pos': np.zeros((20,3),dtype=np.float32),
        }
        self.lock = threading.Lock()
        self.running = False
        self.has_connected = threading.Event()  # 标志是否连接成功
        self.connection_lost = threading.Event()  # 标志连接是否断开

    def run(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.host, self.port))
            self.has_connected.set()  # 标记连接成功
            self.running = True
            print("连接成功")
        except Exception as e:
            print(f"连接失败: {e}")
            self.connection_lost.set()  # 立即标记断开状态
            return

        start = time.time()
        try:
            while self.running:
                # 接收数据
                try:
                    raw_msglen = self._recv_n_bytes(4)
                    if not raw_msglen:
                        return None
                    msglen = int.from_bytes(raw_msglen, byteorder='big')

                    # 接收完整的数据包
                    data = self._recv_n_bytes(msglen)
                    if data is None:
                        print("连接已关闭")
                        break
                    data_dict = pickle.loads(data)

                    # 更新共享变量
                    with self.lock:
                        self.data_dict = data_dict

                    end = time.time()
                    print("接收到数据，耗时：", (end - start))
                    start = time.time()
                except socket.error:
                    print("接收数据时发生错误")
                    break
        except Exception as e:
            print(f"运行时发生错误: {e}")
        finally:
            self.stop()
    def _recv_n_bytes(self, n):
        """
        Helper method to ensure n bytes are received
        """
        data = b""
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def stop(self):
        self.running = False
        self.connection_lost.set()  # 标记连接断开
        if self.client_socket:
            self.client_socket.close()
        print("连接已断开")

    def get_data_dict(self):
        with self.lock:
            return self.data_dict

    def is_connected(self):
        # 检测连接状态
        return self.has_connected.is_set() and not self.connection_lost.is_set()




if __name__ == "__main__":

    receiver = MocapReceiver('192.168.1.167', 12345)
    receiver_thread = threading.Thread(target=receiver.run)
    receiver_thread.start()

    while True:
        time.sleep(0.01)
        data_dict = receiver.get_data_dict()
        if data_dict is not None:
            print("接收到数据")
        # print(data_dict.keys())


    receiver.stop()
    receiver_thread.join()

