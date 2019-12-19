# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 01:49:01 2019

@author: harry
"""

import socket
import cv2
import numpy
import json


class ServerCam:
    def __init__(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(ip)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TCP_IP = '127.0.0.1'
        self.TCP_PORT = 8080
        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.s.listen(True)
        self.conn , self.addr = self.s.accept()
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        self.img_count = []
        
    def recvall(self,sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def get_img(self):
        length     = self.recvall(self.conn,16)           #get img length
        stringData = self.recvall(self.conn, int(length)) #get img
        self.img_count.append(int(self.recvall(self.conn,16)))#append get img count
        
        data = numpy.fromstring(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1)
        return decimg
    
    def send_img(self,img):
        result, imgencode = cv2.imencode('.jpg', img, self.encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
        le = str(len(stringData))
        self.conn.send( str.encode( le.ljust(16)));
        self.conn.send( stringData );

        #self.conn.send(str.encode(str(self.img_count).ljust(16)));
        
    def send_jeson(self,data):
        if len(self.img_count) > 0:
            self.conn.send(str.encode(str(len(str(data))).ljust(16)));
            self.conn.send(str.encode(str(data)));
            self.conn.send(str.encode(str(self.img_count[0]).ljust(16)));
            del self.img_count[0]
            cv2.waitKey(10)
    
    def end(self):
        print("end")
        self.conn.send(str.encode(str(len(str("end"))).ljust(16)));
        self.conn.close()
        cv2.destroyAllWindows()
        
    def start(self):
        #新增cv2視窗.....
        print("start")
        self.conn.send(str.encode(str(len(str("start"))).ljust(16)));
        self.conn.send(str.encode("start"))
        
    def stop(self):
        #暫停cv2視窗...
        print("stop")
        self.conn.send(str.encode(str(len(str("stop"))).ljust(16)));
        self.conn.send(str.encode("stop"))
#       
#cam = Server_cam()
#while 1:
#    try:
#        img = cam.get_img()
#        cam.send_str([20],[20],[100],[100])
#    except ConnectionResetError as e:
#        print("該客戶機異常！已被強迫斷開連接",e)
#        print("正在等待連接")
#        
#        cam.conn , cam.addr = cam.s.accept()
        