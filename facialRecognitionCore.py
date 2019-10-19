# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 02:09:59 2019

@author: Frank-NB
"""
import os
import cv2
import time

import backbone.facenet as facenet

import tensorflow as tf
import numpy as np
from scipy import misc

class PassingInfo:
    def __init__(self ,name ,passRate = 0 ,passAvgDist = 0):
        self.name = name
        self.passRate = passRate
        self.passAvgDist  = passAvgDist
        self.count = 0
        
        #------------------計算---------------------
        self.precision = 100
        self.max       = 10
        self.leastNum  = self.max * 0.3
        self.leastRate = 150 #0.7 * 5 
        self.calInfoList  = []

    def __str__(self):
        if self.name == 'noMan':
            return "noMan: unknow  avg: unknow  min: unknow"
        else:
            return "{0:5}:{1:<2d}  avg: {2:.5f}".format(self.name ,self.passRate ,self.passAvgDist)

    def addInfo(self ,info):
        self.calInfoList.append(info)
    
    def calResult(self):
        if self.calInfoList != []:
            self.calInfoList = sorted(self.calInfoList , key = lambda x : x[1]) #小到大排序

            print("In facialRec:\n" ,"-"*30)
            tempDict = {}
            for i in range(0 ,self.max):
                if i < len(self.calInfoList):
                    name = self.calInfoList[i][0]
                    dist = self.calInfoList[i][1]
                    rate = self.precision - int(dist * self.precision)
                    print("{0} {1}".format(self.calInfoList[i] ,str(rate)))

                    if name not in tempDict:
                        tempDict[name] = [0 ,0 ,0]

                    tempDict[name][0] += rate
                    tempDict[name][1] += dist
                    tempDict[name][2] += 1
                else:
                    break
            for k ,v in tempDict.items():
                print("{0} {1}".format(k ,v))
                if v[0] >= self.passRate and v[0] >= self.leastRate and v[2] >= self.leastNum:
                    self.name         = k
                    self.passRate     = v[0]
                    self.passAvgDist  = v[1]
                    self.count        = v[2]

            if self.count > 0:
                self.passAvgDist /= self.count
            self.calInfoList.clear()  

        print(f"recognitionResult: {self.__str__()} \n {'-'*30}")                 


class Identity:
    def __init__(self ,name ,embList =[]):
        self.name = name
        self.embList = embList  

    def calDistInfo(self ,detectEmb ,distTresh):
        distList = []
        for emb in self.embList:
            dist = np.sqrt(np.sum(np.square(detectEmb - emb)))
            if dist <= distTresh:
                distList.append(dist) 

        return distList

class FacialRecognitionCore:
    def __init__(self):
        #------------變數---------------------------------------------
        self.embedding_size = None
        self.embeddings = None
        self.images_placeholder = None
        self.phase_train_placeholder = None
        self.fileRootPath  = "./Image"#圖片儲存位置                   
        self.image_size = 160
        self.identityList = []        #IdentityList()  
        self.distTresh = 0.85
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        
       
    def initFacenet(self ,modeldir):#FaceNet模型載入
        print('建立facenet embedding模型')
        tf.Graph().as_default()
        
        facenet.load_model(modeldir)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        
        print('facenet embedding模型建立完成')
        
    def initFolderImage(self):      #初始化(計算資料庫的特徵)
        for folder in os.listdir(self.fileRootPath):
            if os.path.isfile("{0}/{1}".format(self.fileRootPath ,folder)):
                continue

            embFileName = "{0}/{1}/{2}".format(self.fileRootPath ,folder ,'image.yf')
            identity = Identity(folder ,[])   
            try:                                
                fileRead = open(embFileName, 'r', encoding='UTF-8')                                                                         
                emb = []                     
                for line in fileRead:
                    itemList = line.split()
                    if len(itemList) == 1:                
                        if len(emb) > 0:
                            identity.embList.append(emb)
                            emb = []
                    else:                          
                        emb.extend( [np.float32(item.strip('[]')) for item in itemList if(item != '[' and item != ']' )] )
                        
                identity.embList.append(emb)
                identity.embList = np.array(identity.embList)
                fileRead.close() 
                print("{0}/{1}/{2}  完成讀檔!!!(num:{3:2d})".format(self.fileRootPath ,folder ,'image.yf' ,len(identity.embList) ))
            
            except FileNotFoundError:
                print("{0}  並不存在!!! 重新產生檔案...".format(embFileName))    
                fileWrite = open(embFileName, 'w' ,encoding='UTF-8') #直接開啟一個檔案，如果檔案不存在則建立檔案
                imgList=[]
                mypath = "{0}/{1}".format(self.fileRootPath ,folder)
                for imageFile in os.listdir(mypath):                 #取得所有檔案與子目錄名稱
                    if os.path.isfile("{0}/{1}".format(mypath ,imageFile)) and imageFile.endswith('.jpg'):
                        print("{0}/{1}".format(mypath ,imageFile))
                        img = cv2.imdecode(np.fromfile("{0}/{1}".format(mypath ,imageFile) ,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                        imgList.append(img)
   
                identity.embList = self.faceCaculate(imgList)        #這一個人的10張照片的emb
                
                for embName ,emb in zip(range(10) ,identity.embList):
                    fileWrite.write("{0}\n{1}\n".format(embName ,emb))
                    
                fileWrite.close()           
                print("{0}/{1}/{2}  完成寫檔!!!(num:{3:2d})".format(self.fileRootPath ,folder ,'image.yf' ,len(identity.embList) ))
                
            self.identityList.append(identity)
               
    def faceCaculate(self ,faceImageList):#計算人臉的特徵
        preProcessingFaceImageList  = []
        emb   = np.array([])
        for faceImage in faceImageList:
            faceImage = cv2.resize(faceImage, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            faceImage = facenet.prewhiten(faceImage)  
            preProcessingFaceImageList.append(faceImage)

        if len(preProcessingFaceImageList) > 0:
            preProcessingFaceImageList = np.array(preProcessingFaceImageList)            
            emb = self.sess.run(self.embeddings ,feed_dict = { self.images_placeholder: preProcessingFaceImageList 
                                                              ,self.phase_train_placeholder:False })     
        return emb
                   
    def compareFace(self ,faceImageList):#辨識圖片
        possiblePassInfoList = []
        if len(faceImageList) > 0 and len(self.identityList) > 0:
            
            detectEmbList = self.faceCaculate(faceImageList)            
            for detectEmb in detectEmbList:   

                possiblePassInfo = PassingInfo('noMan')
                for identity in self.identityList:
                    distList = identity.calDistInfo(detectEmb ,self.distTresh)  
                    for dist in distList:
                        possiblePassInfo.addInfo( (identity.name ,dist) )
                
                possiblePassInfo.calResult()    #計算結果
                possiblePassInfoList.append(possiblePassInfo)  

            #print('-'*30)         
        return possiblePassInfoList