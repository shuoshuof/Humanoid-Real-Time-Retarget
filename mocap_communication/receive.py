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

def receive_data():

    # 配置服务器地址和端口
    HOST = '192.168.1.167'  # 替换为服务器主机的局域网 IP 地址
    PORT = 12345          # 与服务器端使用相同的端口号

    # 创建套接字并连接到服务器
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    try:
        while True:
            # 接收数据
            while True:
                start = time.time()
                packet = client_socket.recv(4096)  # 一次接收 4KB 数据
                if not packet:
                    break
                body_pose = pickle.loads(packet)
                end = time.time()
                print("接收到数据，耗时：", (end - start))

    except KeyboardInterrupt:
        print("客户端关闭连接")
    finally:
        # 关闭连接
        client_socket.close()

if __name__ == "__main__":
    receive_data()


