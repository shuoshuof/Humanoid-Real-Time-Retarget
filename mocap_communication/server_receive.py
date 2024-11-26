# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 11:49
@Auth ： shuoshuof
@File ：server_receive.py
@Project ：Humanoid-Real-Time-Retarget
"""

import socket
import pickle
import time


class Receiver:
    def __init__(self, host='192.168.1.13', port=7000):
        self.host = host
        self.port = port
        self.buffer_size = 8192  # 增大缓冲区
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"已连接到服务器 ({self.host}:{self.port})")

    def receive(self):
        # 接收数据的总长度
        raw_msglen = self._recv_n_bytes(4)
        if not raw_msglen:
            return None
        msglen = int.from_bytes(raw_msglen, byteorder='big')

        # 接收完整的数据包
        data = self._recv_n_bytes(msglen)
        if data is None:
            return None

        # 反序列化数据
        received_data = pickle.loads(data)
        return received_data

    def _recv_n_bytes(self, n):
        """
        Helper method to ensure n bytes are received
        """
        data = b""
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data


if __name__ == '__main__':
    receiver = Receiver('192.168.1.157', 37535)
    receiver.connect()
    while True:
        received_dict = receiver.receive()
        if received_dict is not None:
            # print(received_dict)
            print("Received:", received_dict['dof_pos'].shape)
            time.sleep(0.001)