# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 01:50:04 2019

@author: harry
"""

import socket
import cv2
import numpy
import json

class ClientCam:
    def __init__(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(ip)
        self.s = socket.socket()
        self.TCP_IP = ip #'127.0.0.1'
        self.TCP_PORT = 8080#11794
        #self.capture = cv2.VideoCapture(0)
        self.s.connect((self.TCP_IP, self.TCP_PORT))
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        #self.ret, self.frame = self.capture.read()
        self.sned_q = []
        self.get_q = []
        self.state = "stop"
        self.img_count = 0
        self.get_img_count = 0

    def recvall(self,sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def send_img(self ,frame):
        #self.ret, self.frame = self.capture.read()
        
        result, imgencode = cv2.imencode('.jpg', frame, self.encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
        le = str(len(stringData))
        
        self.s.send( str.encode( le.ljust(16)));
        self.s.send(stringData);
        self.s.send(str.encode(str(self.img_count).ljust(16)));
        self.sned_q.append((self.img_count ,frame))
        self.img_count += 1
          
    def get_data(self):
        length = self.recvall(self.s,16)
        print(length)
        jsonData = self.recvall(self.s, int(length))
        if jsonData == b'stop':
            print("STOP")
            self.state = "stop"
            self.sned_q.clear()
            return
        elif jsonData == b'start':
            print("START")
            self.state = "start"
            return
        elif jsonData == b'end':
            print("START")
            self.state = "end"
            return
        self.get_img_count = int(self.recvall(self.s,16))
        return jsonData
    
    def get_img(self):   
       '''
       length = self.recvall(self.s,16)
       stringData = self.recvall(self.s, int(length))
       if stringData == b'stop':
           print("STOP")
           self.state = "stop"
           self.q.clear()
           return
       elif stringData == b'start':
           print("START")
           self.state = "start"
           return
       elif stringData == b'end':
           print("START")
           self.state = "end"
           return
       #self.get_img_count = int(self.recvall(self.s,16))
       '''
       length = self.recvall(self.s,16)
       stringData = self.recvall(self.s, int(length))
       data = numpy.frombuffer(stringData, dtype='uint8')

       #length = self.recvall(self.s,16)
       #stringData = self.recvall(self.s, int(length))

       decimg = cv2.imdecode(data,1)
       #cv2.imshow("result",decimg)
       #cv2.waitKey(10)
       #print(length, jsonData)
       #return stringData
       return decimg
        
    def show(self,data):
        print(data)
        
        if self.state == "start":
            text = json.loads(data)
            temp = self.sned_q[0]
            print("Location : {} Server : {} Locationshow : {}".format(self.img_count,self.get_img_count,temp[0]))
            img = temp[1]
            img = cv2.flip(img,1)
            for l in text:
                cv2.rectangle(img, (int(l[1]), int(l[2])), (int(l[1]+l[3]),int(l[2]+l[4])), (0, 255, 0), 2)
            cv2.putText(img, "deley : {}".format(temp[0]-self.get_img_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("result",img)
            if self.get_img_count >= temp[0]:
                if len(self.sned_q) > 0:
                        del self.sned_q[0]
            cv2.waitKey(10)            
        
    def end(self):
        #self.capture.release()
        self.s.close()
        #cv2.destroyAllWindows()



# cline = Cline_cam()
        
# while True:
#     try:
#         cline.send_img()
#         data = cline.get_data()
#         if data != None :
#             cline.show(data)
#     except Exception as e:
#         print(e)
#         break
    
#    cline.send_img()
#    data = cline.get_data()
#    if data != None :
#        cline.show(data)

# cline.end()
