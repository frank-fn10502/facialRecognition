# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 02:07:42 2019

@author: Frank-NB
"""
#import AlexAB darknet.py form yolo_cpp_dll.dll 
import backbone.darknet as darknet
import os
import cv2

class FaceImageInfo:
    def __init__(self ,detection ,mainImg):                
        self.objName   = detection[0].decode("utf-8")
        self.confScore = detection[1]#round(detection[1] * 100, 2)

        self.calBBXShape(detection ,mainImg)
        '''
        x, y, w, h = detection[2][0] ,detection[2][1] ,detection[2][2] ,detection[2][3]
        self.yolo_x = x
        self.yolo_y = y
        self.w      = w
        self.h      = h

        self.xmin  = int(round(x - (w / 2)))    #convertBack
        self.xmax  = int(round(x + (w / 2)))
        self.ymin  = int(round(y - (h / 2)))
        self.ymax  = int(round(y + (h / 2)))
        self.centroid = (self.xmin + w / 2 ,self.ymin + h / 2) #計算重心(形心)
        
        self.__confirmInImage(mainImg)
        '''
      
    def __confirmInImage(self ,mainImg):
        height, width, channels = mainImg.shape
        self.xmin = self.xmin if self.xmin > 0       else 0
        self.ymin = self.ymin if self.ymin > 0       else 0
        self.xmax = self.xmax if self.xmax < width  else width
        self.ymax = self.ymax if self.ymax < height else height

    def calBBXShape(self ,detection ,mainImg):
        x, y, w, h = detection[2][0] ,detection[2][1] ,detection[2][2] ,detection[2][3]
        self.yolo_x = x
        self.yolo_y = y
        self.w      = w
        self.h      = h

        self.xmin  = int(round(x - (w / 2)))    #convertBack
        self.xmax  = int(round(x + (w / 2)))
        self.ymin  = int(round(y - (h / 2)))
        self.ymax  = int(round(y + (h / 2)))
        self.centroid = (self.xmin + w / 2 ,self.ymin + h / 2) #計算重心(形心)
        
        self.__confirmInImage(mainImg)

    
class FacialDectionCore:
    def __init__(self):        
        #----------------所需變數---------------------------------------
        self.netMain  = None
        self.metaMain = None
        self.altNames = None
        self.darknet_image = None
        self.yoloTresh = 0.7
        self.minimalBBX = (20, 20)

    '''
    def initYOLO(self ,configPath = "./cfg/yolov3-tiny-gender-test.cfg"
                      ,weightPath = "./weights/yolov3-tiny-gender-newAnchors_50000.weights"
                      ,metaPath = "./cfg/obj_gender.data"):   
    '''  
    def initYOLO(self ,configPath = "./cfg/yolov3-tiny-face.cfg"
                    ,weightPath = "./weights/yolov3-tiny-face_55000.weights"
                    ,metaPath = "./cfg/obj.data"):
                    
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `"    + os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `"    + os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode("ascii") ,weightPath.encode("ascii") ,0 ,1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents ,re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass       
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain) 
                                               ,darknet.network_height(self.netMain),3)   #Create an image we reuse for each detect   

    def captureData(self ,frame_read):
        frame_rgb     = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                  (darknet.network_width(self.netMain),
                                   darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())    
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=self.yoloTresh)
        
        faceImageInfoList = []        
        for detection in detections:
            if detection[2][2] >= self.minimalBBX[0] or detection[2][2] >= self.minimalBBX[1]:          
                faceImageInfoList.append( FaceImageInfo(detection ,frame_resized) ) #所有偵測到的 "人臉bounding box" 的資訊   
                print(detection)    

        print(f'image_size: ({frame_resized.shape[1]} x {frame_resized.shape[0]})')
        return frame_resized ,faceImageInfoList