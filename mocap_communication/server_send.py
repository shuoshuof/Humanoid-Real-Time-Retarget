# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 11:49
@Auth ： shuoshuof
@File ：server_send.py
@Project ：Humanoid-Real-Time-Retarget
"""
import socket
import pickle

class Transmitter:
    def __init__(self,host='192.168.1.13',port=7000):
        self.host = host
        self.port = port
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host,self.port))
        self.sock.listen(1)

        print(f"服务器已启动，等待连接 ({self.host}:{self.port})...")
        self.conn, self.addr = self.sock.accept()
        print(f"连接已建立：{self.addr}")
    def send(self, data_dict):
        try:
            # 序列化字典为字节流
            data = pickle.dumps(data_dict)
            # 在数据前附加长度信息
            data_with_len = len(data).to_bytes(4, byteorder='big') + data
            # 发送字节流
            self.conn.sendall(data_with_len)
            print(f"字典发送成功，大小：{len(data)} 字节")
        except Exception as e:
            print(f"发送字典时出错：{e}")

if __name__ == '__main__':
    import numpy as np
    import time
    transmitter = Transmitter('192.168.1.13', 7000)
    transmitter.connect()
    dof_state = np.zeros(30)
    img = np.zeros((3,224,224,3),dtype=np.uint8)
    while True:
        transmitter.send({'dof_state': dof_state, 'img': img})
        time.sleep(0.001)