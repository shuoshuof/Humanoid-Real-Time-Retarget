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


import socket
import pickle
import threading
import time

class MocapReceiver:
    def __init__(self, host, port, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.client_socket = None
        self.body_pose = np.zeros((32, 3))
        self.lock = threading.Lock()
        self.running = False
        self.has_connected = threading.Event()  # 标志是否连接成功
        self.connection_lost = threading.Event()  # 标志连接是否断开

    def run(self):
        # 创建套接字并连接到服务器
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

        try:
            while self.running:
                # 接收数据
                start = time.time()
                try:
                    packet = self.client_socket.recv(self.buffer_size)
                    if not packet:
                        print("连接已关闭")
                        break
                    body_pose = pickle.loads(packet)
                    end = time.time()

                    # 更新共享变量
                    with self.lock:
                        self.body_pose = body_pose

                    # print("接收到数据，耗时：", (end - start))
                except socket.error:
                    print("接收数据时发生错误")
                    break
        except Exception as e:
            print(f"运行时发生错误: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.connection_lost.set()  # 标记连接断开
        if self.client_socket:
            self.client_socket.close()
        print("连接已断开")

    def get_body_pose(self):
        with self.lock:
            return self.body_pose

    def is_connected(self):
        # 检测连接状态
        return self.has_connected.is_set() and not self.connection_lost.is_set()




if __name__ == "__main__":
    receiver = MocapReceiver('192.168.1.167', 12345)
    receiver.run()


