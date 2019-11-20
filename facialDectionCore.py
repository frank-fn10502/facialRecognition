# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 02:07:42 2019

@author: Frank-NB
"""
#import AlexAB darknet.py form yolo_cpp_dll.dll 
import backbone.darknet as darknet
import os
import cv2

from utils import Dot ,Square  ,FacialFeature

class YoloPred:
    def __init__(self ,detection):
        self.x, self.y, self.w , self.h = \
        detection[2][0] ,detection[2][1] ,detection[2][2] ,detection[2][3]

        self.objName   = detection[0].decode("utf-8")
        self.confScore = detection[1]#round(detection[1] * 100, 2)

class BBX:
    yoloImgSize = None  #Square(w ,h) 
    oriImgSize  = None  #Square(w ,h) 
    def __init__(self ,yoloPred):
        #------------------設定-----------------------------------------     
        self.fullFaceOffest = 30
        self.fullFace = True  #確定臉沒有被邊緣切割掉
        #------------------變數-----------------------------------------
        self.yolo = yoloPred
        self.centroid = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        self.__calBBXShape() 
        self.__confirmInImage()

    def __calBBXShape(self):
        self.xmin  = int(round(self.yolo.x - (self.yolo.w / 2)))    #convertBack
        self.xmax  = int(round(self.yolo.x + (self.yolo.w / 2)))
        self.ymin  = int(round(self.yolo.y - (self.yolo.h / 2)))
        self.ymax  = int(round(self.yolo.y + (self.yolo.h / 2)))
        self.centroid = Dot(self.yolo.x ,self.yolo.y) #計算重心(形心)       

    def __confirmInImage(self):
        pre_dot_list = self.xmin ,self.ymin ,self.xmax ,self.ymax

        self.xmin = self.xmin if self.xmin > 0       else 0
        self.ymin = self.ymin if self.ymin > 0       else 0
        self.xmax = self.xmax if self.xmax < BBX.yoloImgSize.w else BBX.yoloImgSize.w
        self.ymax = self.ymax if self.ymax < BBX.yoloImgSize.h else BBX.yoloImgSize.h
        
        pos_dot_list = self.xmin ,self.ymin ,self.xmax ,self.ymax

        for now ,pre in zip(pos_dot_list ,pre_dot_list):
            if abs(now - pre) > self.fullFaceOffest:
                self.fullFace = False

    def reCal(self ,yoloPred ,yolo_w ,yolo_h):
        self.yolo = yoloPred
        self.yolo.w = yolo_w
        self.yolo.h = yolo_h
        self.__calBBXShape() 
        self.__confirmInImage()

    
class FacialDectionCore:
    def __init__(self):        
        #----------------所需變數---------------------------------------
        self.netMain  = None
        self.metaMain = None
        self.altNames = None
        self.darknet_image = None

        #----------------屬性-------------------------------------------
        self.mainImgSize = None #(W ,H)
        self.yoloTresh = 0.7
        self.minimalBBX = Square(30, 30)

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

        self.mainImgSize = Square(darknet.network_width(self.netMain) ,darknet.network_height(self.netMain))
        BBX.yoloImgSize = self.mainImgSize

    def captureData(self ,frame_read):
        frame_resized = cv2.resize(frame_read,
                                   self.mainImgSize.getTuple(),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())    
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=self.yoloTresh)
        
        facialFeatureList = []        
        for detection in detections:
            yolo = YoloPred(detection)
            qualifiedFace =  yolo.w >= self.minimalBBX.w and yolo.h >= self.minimalBBX.h   
            
            ff = FacialFeature(qualifiedFace)
            ff.bbx = BBX(yolo)
            facialFeatureList.append(ff) #所有偵測到的 "人臉bounding box" 的資訊   
            #print(detection)    

        #print(f'image_size: ({frame_resized.shape[1]} x {frame_resized.shape[0]})')
        return frame_resized ,facialFeatureList