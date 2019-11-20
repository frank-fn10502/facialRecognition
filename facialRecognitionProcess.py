import sys
import time
import datetime
import os
import numpy as np
import math
import threading
import json
import configparser as cfg
import cv2
import socket
import warnings
import requests


import backbone.darknet as darknet

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt#####

import facialDectionCore
import facialRecognitionCore
from facialRecognitionCore import Identity
import ageGenderCore
import emotionCore
from utils import CircularQueue ,OutputHandler ,Dot ,Square  ,FacialFeature 


class AccData:  #Accumulated recognized facial data
    def __init__(self ,facialFeature):
        #-------------------參數----------------------------
        self.dataList = CircularQueue(10)     #[(name1 ,info1) ,(name2 ,info2) , ...]

        self.treshDisplay = 0.6
        self.treshIOU = 0.8
        self.num_counter = 0
        self.counter  = 5         #設定存活的次數(如果都偵測不到)
        

        #-------------------變數---------------------------- 
        self.lastFacialFeature = None

        self.qualifiedFace = True
        self.bbx      = facialFeature.bbx
        self.name     = facialFeature.identity.name if facialFeature.identity is not None else None
        self.gender   = facialFeature.gender
        self.age      = facialFeature.age     
        self.emotion  = facialFeature.emotion #####
              
        self.addInfo(facialFeature)

    def __str__(self):
        return f"{self.name if self.name is not None else 'noMan'} ({self.gender.getGenderName()},{round(self.age)})"

    def __calIOU(self ,anotherFacialFeature):
        #如果有交集才計算
        if(abs(self.lastFacialFeature.bbx.centroid.x - anotherFacialFeature.bbx.centroid.x) < (self.lastFacialFeature.bbx.yolo.w + anotherFacialFeature.bbx.yolo.w) / 2 and
           abs(self.lastFacialFeature.bbx.centroid.y - anotherFacialFeature.bbx.centroid.y) < (self.lastFacialFeature.bbx.yolo.h + anotherFacialFeature.bbx.yolo.h) / 2):
           inter_x1 = max(self.lastFacialFeature.bbx.xmin ,anotherFacialFeature.bbx.xmin)
           inter_y1 = max(self.lastFacialFeature.bbx.ymin ,anotherFacialFeature.bbx.ymin)
           inter_x2 = min(self.lastFacialFeature.bbx.xmax ,anotherFacialFeature.bbx.xmax)
           inter_y2 = min(self.lastFacialFeature.bbx.ymax ,anotherFacialFeature.bbx.ymax)
           inter_square = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
           union_square = self.lastFacialFeature.bbx.yolo.w * self.lastFacialFeature.bbx.yolo.h + anotherFacialFeature.bbx.yolo.w   * anotherFacialFeature.bbx.yolo.h   - inter_square

           IOU = inter_square / union_square * 1.0

           return IOU >= self.treshIOU ,IOU
        else:
            return False ,0

    def addInfo(self ,facialFeature):
        self.counter = 5
        self.lastFacialFeature = facialFeature
        
        self.dataList.push(facialFeature)
        self.__adjustBBX(facialFeature)
        
    def isThisAccData(self ,faceImgInfo):
        return self.__calIOU(faceImgInfo)

    def __adjustBBX(self ,facialFeature):
        w = 0
        h = 0
        count = self.dataList.len()
        for data in self.dataList.queue:
            if data != '':
                w += data.bbx.yolo.w
                h += data.bbx.yolo.h

        w /= count
        h /= count
        self.bbx.reCal(facialFeature.bbx.yolo ,math.floor(w) ,math.floor(h))     

    def isStillAlive(self):
        self.counter -= 1
        return self.counter >= 0

    def calResult(self):
        nameDict = {}
        genderList = [0 ,0 ,0] #none ,male ,female
        ageAvg = 0

        self.qualifiedFace = True
        for f in self.dataList.getList():
            if not f.qualifiedFace:
                self.qualifiedFace = False
                break

     
        if self.qualifiedFace:
            for f in self.dataList.getList():
                genderList[f.gender.genderIndex + 1] += 1

                if f.identity.name is not None:
                    if f.identity.name in nameDict: 
                        nameDict[f.identity.name] += 1
                    else:
                        nameDict[f.identity.name] = 1

                ageAvg += f.age

            self.name = 'noMan'
            currentMax = 0
            for accName ,count in nameDict.items():
                if count >= currentMax and count >= math.ceil(self.treshDisplay * self.dataList.len()):
                    currentMax = count
                    self.name = accName 

            self.gender = ageGenderCore.Gender(np.argmax(genderList,axis=0) - 1)
            self.age    = ageAvg / self.dataList.len()   

            print(f"name:{nameDict}  ,gender:{genderList}  ,age:{self.age :.2f} ,num:{self.dataList.len()}")  


class Recognition:
    def __init__(self):        
        # --------------------------設定人臉偵測演算法------------------------------------
        self.facialDetection = facialDectionCore.FacialDectionCore()  # 讀取並設定YOLO
        # ------------------------------------設定人臉辨識演算法--------------------------
        self.facialRecognition = facialRecognitionCore.FacialRecognitionCore()  

        #------------初始化mobilenet模型和資料-gender---------------------------
        self.genderAgeRec = ageGenderCore.GenderAgeRec()

        #------------初始化模型和資料-emotion----------------------------------       
        self.emotionRec = emotionCore.EmotionRec()

        #---------------OPENCV---------------------------------------------
        self.capDevice = 0 #'http://admin:123456@140.126.20.95/video.cgi'#0  # "./data/YouTube_Rewind_2016.mp4"
        self.deviceW = 1920
        self.deviceH = 1080
        self.cap = None
        self.sock = socket.socket()#遠端攝影機
        

        #--------------參數----------------------------------------------
        self.jsonFilePath = './other/outputData/detectionData.json'
        self.writeJson    = False#True
        self.maxFaceCount = 10
        self.saveInterval = 0.5
        self.cfgFile = './cfg/myConfig.cfg'
        
        self.colorList = []
        self.possiblePassInfoList = []
        #----------------------------------------------------------------
        self.resultList = None
        self.facialFeatureList = None
        self.currentImage = None
        self.resizeImg    = None

        self.accDataList     = []               #累計的人臉圖片(用以穩定bbx && 觀察辨識結果,做修正)
        self.displayTextList = []               #需要顯示的文字串

        self.pre_time = 0

        self.faceName = 'noMan'
        self.regPreTime = 0       
        self.currentRegImgList = []
        self.faceitem = 0
        self.needName = True
        self.regDone  = False

        self.outputHandler = OutputHandler()

    def initAllData(self):
        #----------------讀取cfg資料----------------------------------
        if not self.__initCfgData():
            return False
        
        pre_t = time.time()
        #----------------初始化yolo模型-------------------------------
        t1 = threading.Thread( target=self.facialDetection.initYOLO )

        #------------初始化facenet模型和資料---------------------------
        t1.start()
        self.facialRecognition.initFacenet("./weights/20180402-114759.pb")# 初始化facenet模型和資料 20180402-114759.pb 20180408-102900.pb       
        self.facialRecognition.initFolderImage()

        #------------初始gender_age-----------------------------------  
        self.genderAgeRec.initModel()

        #------------初始gender_age-----------------------------------  
        self.emotionRec.initModel()

        #------------初始攝影機參數------------------------------------    
        self.__initCapDevice()

        t1.join()

        print(f'perpare all cost:{time.time() - pre_t:.3f}')
        return True

    def startCap(self):#server的start,stop需要同時加進來  #server的end 會告訴我們server掛了
        self.stopThread = False
        threading.Thread(target=self.__getImge).start()
        self.outputHandler.start()
        return self

    def stopCap(self):
        self.stopThread = True

    def isRegDone(self):
        return self.regDone

    def initRegData(self):
        self.faceitem = 0
        self.currentRegImgList = []
        self.needName = True
        self.regDone  = False  
        self.faceName = 'noMan'    

    def setRegName(self ,name):
        self.needName   = False
        self.faceName   = name
        self.regPreTime = time.time()

    def registered(self ,displayInfo = True):
        facialFeatureList ,resultList = self.__preProcess()
        regFaceImg  = None
        regFaceInfo = None
        for info ,result in zip(facialFeatureList ,resultList):
            if ((info.ymin * self.c_h <= self.regAreaH) and (info.ymax * self.c_h >= self.regAreaPt2[1]) and 
                (info.xmin * self.c_w >= self.regAreaW) and (info.xmax * self.c_w <= self.regAreaPt2[0]) ):
                
                croppedImage = self.resizeImg[info.ymin:info.ymax,info.xmin:info.xmax]   # 裁剪座標為[y0:y1, x0:x1]
                croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB)

                regFaceImg  = croppedImage
                regFaceInfo = info

            if result is not None:
                result.addInfo(info)
            else:
                self.accDataList.append(AccData(info))

        if regFaceImg is not None:
            if time.time() - self.regPreTime >= self.saveInterval:
                self.regPreTime = time.time()
                if (self.faceitem == 1 and not self.needName) or (self.faceitem != 1 and not self.regDone):
                    self.faceitem += 1
                    self.currentRegImgList.append(regFaceImg)

            if self.faceitem >= self.maxFaceCount and not self.regDone:   #開始寫檔                        
                if not os.path.exists(self.facialRecognition.fileRootPath + "/" + self.faceName + "/"):
                    os.makedirs(self.facialRecognition.fileRootPath + "/" + self.faceName + "/")

                for regFaceImg ,i in zip( self.currentRegImgList ,range(0 ,10) ):
                    print('儲存照片: {0}{1}.jpg'.format(self.faceName ,i))
                    cv2.imencode('.jpg', regFaceImg)[1].tofile("{0}/{1}/{2}.jpg".format(self.facialRecognition.fileRootPath ,self.faceName ,i))
                print('儲存照片完成')
               
                for identity in self.facialRecognition.identityList:
                    if identity.name == self.faceName:
                        self.facialRecognition.identityList.remove(identity)

                embList = self.facialRecognition.faceCaculate(self.currentRegImgList)        
                self.facialRecognition.identityList.append(Identity(self.faceName, embList))

                self.regDone  = True

            if displayInfo:
                self.displayTextList.append(["完成:{0} %".format(self.faceitem * 10), (regFaceInfo.xmin, regFaceInfo.ymin), 20])
        else:
            if displayInfo:
                self.displayTextList.append(["請靠近一點!!!", (self.regAreaPt1[0], self.regAreaPt1[1]), 16])
                print("不在界線內")

                for info in facialFeatureList:
                    self.displayTextList.append(["完成:{0} %".format(self.faceitem * 10), (info.xmin, info.ymin), 20])

        if displayInfo:
            cv2.rectangle(self.currentImage, self.regAreaPt1, self.regAreaPt2,(255, 0, 0), 3)  #劃出註冊的有效區域 
            colorList = [ ( 0 ,0 ,255) for i in range(len(facialFeatureList))]
            self.__drawOnImg(facialFeatureList ,colorList)
            #self.__drawOnImg(facialFeatureList)

        return self.faceitem == 1 and self.needName ,regFaceImg

    def recognized(self  ,displayInfo = True):
        pre_t = time.time()
        self.__preProcess()
        yolo_time = time.time() - pre_t

        faceImages = []
        faceImage_ori_list = []
        for f in self.facialFeatureList:
            if f.qualifiedFace:
                bbx = f.bbx
                # 裁剪座標為[y0:y1 ,x0:x1]
                croppedImage = self.resizeImg[bbx.ymin:bbx.ymax ,bbx.xmin:bbx.xmax]
                faceImages.append(croppedImage)
                faceImage_ori_list.append(self.__getFacialImgWithPatch(bbx))   
        '''
        temp1 = np.asarray(faceImages[0]).shape
        temp2 = np.asarray(faceImage_ori_list[0]).shape
        if temp1[0] != 0:
            print(f"faceImages: {temp1},比值(h/w):{temp1[1] / (temp1[2] * 1.0) :.3f}\n" + \
                  f"faceImage_ori_list:{temp2} ,比值(h/w):{temp1[1] * self.c_h / (temp1[2] * self.c_w) :.3f}")   
               
        else:
            print(f"faceImages{faceImages} length:{len(faceImages) }")
        '''

        #####
        pre_t = time.time()
        self.genderAgeRec.predict(self.facialFeatureList ,faceImage_ori_list)
        g_time = time.time() - pre_t       
        #####      
        
        #####
        pre_t = time.time() 
        self.facialRecognition.compareFace(self.facialFeatureList ,faceImages)
        id_time = time.time() - pre_t
        #####

        '''情緒暫時不用
        pre_t = time.time()
        self.emotionRec.predict(facialFeatureList ,faceImage_ori_list)
        e_time = time.time() - pre_t    
        '''

        print(f"self.facialFeatureList :{len(self.facialFeatureList)}")
        self.resultList = self.__getResultList(self.facialFeatureList) #從累計的list中取得結果

        if len(self.resultList) > 0:
            for index ,result in enumerate( self.resultList):
                print(f"{index}. " ,end='')
                result.calResult()

        if displayInfo:   
            self.__drawOnImg()

        #print(f"reco gender_age: {g_time:.2f}s | reco id:{id_time:.2f}s | yolo: {yolo_time:.2f} | emotion: {e_time:.2f} | total time: {id_time + g_time + yolo_time + e_time:.4f}")
        #print(f"reco id:{id_time:.2f}s | yolo: {yolo_time:.2f} | total time: {id_time + yolo_time:.4f}") 
        print("=" * 40 ,"\n")  

        #self.postAnswer()
        return self.__createJsonFile(self.facialFeatureList)
 
    def stopAll(self):
        self.outputHandler.write = False
    
    def postAnswer(self):
        my_data = {'msg' : 'f3'}
        for displayText in self.displayTextList:
            data = displayText[0].split(' [')[0]
            if data != 'noMan':
                my_data['msg'] = 'n3'
                break
        
        r = requests.post(' http://localhost/IOT/V2/control/pubControl.php' ,data = my_data)
        

    #-----------------------------private func----------------------------------------------
    def __getFacialImgWithPatch(self ,bbx):
        pt1 = (int(bbx.xmin * self.c_w), int(bbx.ymin * self.c_h))
        pt2 = (int(bbx.xmax * self.c_w), int(bbx.ymax * self.c_h))
        croppedImage = self.currentImage[pt1[1]:pt2[1] ,pt1[0]:pt2[0]]

        h ,w ,c = croppedImage.shape
        border_num = math.ceil(abs(h - w) / 2.0)

        img_patch = None
        if w > h:#上和下
            img_patch = cv2.copyMakeBorder(croppedImage, border_num, border_num, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:    #左和右
            img_patch = cv2.copyMakeBorder(croppedImage,  0, 0, border_num, border_num, cv2.BORDER_CONSTANT, value=(0, 0, 0)) 

        return img_patch      

    def __drawOnImg(self):
        # ---------------------劃出文字---------------------------------------------------
        self.__displayText()

        #----------------------劃出bbx---------------------------------------------------- 
        self. __drawBBX()          

        # ---------------------顯示圖片---------------------------------------------------
        fps = 1 / (time.time() - self.prev_time)
        processTime = time.time() - self.prev_time
        cv2.putText(self.currentImage ,"FPS:{0:.2f}  time:{1:.5f}  faceNum:{2}".format(
                    fps, processTime, len(self.facialFeatureList) ),
                    (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 0, 0], 1)

        #self.outputHandler.addResult(f"{datetime.datetime.now()} {processTime:.4f} {len(facialFeatureList)}")
        print( f"height:{self.currentImage.shape[0]} ,width:{self.currentImage.shape[1]}" \
               f" fps:{fps:.2f} time:{processTime:.5f} faceNum:{len(self.facialFeatureList)}")

    def __createJsonFile(self ,facialFeatureList):
        if facialFeatureList is not None:
            dataList = []
            for displayText ,info in zip(self.displayTextList ,facialFeatureList):
                #[name ,minx ,miny ,w ,h]
                data = [displayText[0].split(' [')[0] ,info.xmin * self.c_w ,info.ymin * self.c_h ,info.w ,info.h] 
                dataList.append(data)

            myData = json.dumps(dataList ,ensure_ascii=False)
            #print(myData)
            if self.writeJson:
                try:
                    with open(self.jsonFilePath ,'w' ,encoding='UTF-8') as jf:
                        jf.write(myData)
                except Exception as e:
                    print(e)                        
        return myData

    def __initCfgData(self):
        config = cfg.ConfigParser()
        try:
            config.read(self.cfgFile ,encoding='utf-8')
            self.capDevice = config['Default']['device'] #???
            self.capDevice = int(self.capDevice) if self.capDevice.isdigit() else self.capDevice
            self.deviceW = config.getint('Default' ,'devicew')
            self.deviceH = config.getint('Default' ,'deviceh')

            self.maxFaceCount = config.getint('Default' ,'maxfacenum')
            self.saveinterval = config.getfloat('Default' ,'saveinterval')
            self.regAreaW     = config.getint('Default' ,'regAreaw')    #-1 代表預設
            self.regAreaH     = config.getint('Default' ,'regAreah')

            self.jsonFilePath = config['Default']['jsonfilepath']
            self.writeJson    = config.getboolean('Default' ,'writejson')

            self.facialDetection.yoloTresh = config.getfloat('DectionCore' ,'thresh')

            self.facialRecognition.distTresh = config.getfloat('RecognitionCore' ,'disttresh')
            self.facialRecognition.fileRootPath = config.get('RecognitionCore' ,'imagefolder')

            self.__insureHasFolder('decDataResultPath' ,self.jsonFilePath)
            self.__insureHasFolder('imagePath' ,self.facialRecognition.fileRootPath)

        except FileNotFoundError:
            print('你的cfg檔案好像不見了!!!')

        except Exception as e:
            print(e)
            return False

        return True

    def __insureHasFolder(self ,name ,path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'{name}: {self.jsonFilePath}')

    def __initCapDevice(self):
        if self.capDevice != '-1' :
            self.cap = cv2.VideoCapture(self.capDevice + cv2.CAP_DSHOW)
            print( 'capture is {}{}\n'.format(self.cap.isOpened() ,'' if self.cap.isOpened() else ' \n\t please check the camera is existed or recapture again.') )
            if self.cap.isOpened():        
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH ,self.deviceW )
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.deviceH )

                self.capHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.capWidth  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.regAreaW = int(self.capWidth/4)  if self.regAreaW == -1 else int((self.capWidth  - self.regAreaW)/2)
                self.regAreaH = int(self.capHeight/4) if self.regAreaH == -1 else int((self.capHeight - self.regAreaH)/2)
                print(f'regAreaW: {self.regAreaW} ,regAreaH:{self.regAreaH}')

                self.regAreaPt1 = (self.regAreaW, self.regAreaH )
                self.regAreaPt2 = (self.capWidth - self.regAreaW ,self.capHeight - self.regAreaH)
                self.c_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  / darknet.network_width(self.facialDetection.netMain)
                self.c_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / darknet.network_height(self.facialDetection.netMain) 

                self.ret ,self.frame = self.cap.read()  #初始化
                self.stopThread = False   
        else:
                #cline的get_img可以取代
                self.cap = None
                self.stopThread = False
                TCP_IP = "192.168.100.143"
                TCP_PORT = 8080
                self.sock.connect((TCP_IP, TCP_PORT))
                length = self.recvall(self.sock,16)
                stringData =self.recvall(self.sock, int(length))
                data = np.fromstring(stringData, dtype='uint8')
                decimg=cv2.imdecode(data,1)
                sp = decimg.shape

                #以下cfg
                self.capHeight = int(sp[0])
                self.capWidth  = int(sp[1])
                self.regAreaW = int(self.capWidth/4)  if self.regAreaW == -1 else int((self.capWidth  - self.regAreaW)/2)
                self.regAreaH = int(self.capHeight/4) if self.regAreaH == -1 else int((self.capHeight - self.regAreaH)/2)
                print(f'regAreaW: {self.regAreaW} ,regAreaH:{self.regAreaH}')

                self.regAreaPt1 = (self.regAreaW, self.regAreaH )
                self.regAreaPt2 = (self.capWidth - self.regAreaW ,self.capHeight - self.regAreaH)
                self.ret = True
                self.frame = decimg
                self.c_w = sp[1]/ darknet.network_width(self.facialDetection.netMain)
                self.c_h = sp[0]/ darknet.network_height(self.facialDetection.netMain)
  
    def __getImge(self):
        '''
        while True:
            if self.stopThread:
                return
            
            self.ret ,self.frame = self.cap.read()
            time.sleep(0.016)
        '''
        while True:
            if self.stopThread:
                return

            if self.cap == None:
                length = self.recvall(self.sock,16)
                stringData =self.recvall(self.sock, int(length))
                
                data = np.fromstring(stringData, dtype='uint8')
                decimg=cv2.imdecode(data,1)
                
                self.ret = True
                self.frame = decimg
                cv2.waitKey(5)
            else:
                self.ret ,self.frame = self.cap.read()
                time.sleep(0.016)

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def __getResultList(self, facialFeatureList): #用累計的datalist推測出結果
        resultList = []                         #對應到相對的 累計list(用這list推測出結果)
        for index ,f in enumerate(facialFeatureList):           
            hasAcc = False
            for accData in self.accDataList:
                isPass ,IOU = accData.isThisAccData(f)
                if isPass:        
                    accData.addInfo(f)#####
                    resultList.append(accData)
                    hasAcc = True
                    break
            
            print(f"{index}. iou: {IOU if hasAcc else 'noMatch'}")
            if not hasAcc:
                result = AccData(f)
                self.accDataList.append(result)
                resultList.append(result)#####
                #print("iou:{0}".format('noMatch') )
        print()
                
        
        #------------移除掉很久沒新增過的 accResult-------------
        for accData in self.accDataList:                  
            if not accData.isStillAlive():
                self.accDataList.remove(accData)
        #-----------------------------------------------------
        return resultList
    
    def __preProcess(self):
        self.displayTextList.clear()
        self.currentImage = cv2.flip(self.frame ,1)  #左右翻轉
        self.currentImage = cv2.cvtColor(self.currentImage, cv2.COLOR_BGR2RGB)

        self.resizeImg ,self.facialFeatureList = self.facialDetection.captureData(self.currentImage)  #currentImage的畫面 ,從yolo網路中取得bbx的參數(存成list)

    def __displayText(self):
        fontSize = 20
        color    = (255, 0, 0)
        imgPIL   = Image.fromarray(self.currentImage)
        draw     = ImageDraw.Draw(imgPIL)
        font     = ImageFont.truetype('simsun.ttc', fontSize)

        for result in self.resultList:     
            if result.qualifiedFace:
                draw.text((result.bbx.xmin * self.c_w ,result.bbx.ymin * self.c_h - 20), str(result), font=font, fill=color)
              
        self.currentImage = np.asarray(imgPIL)

    def __drawBBX(self):
        for result in self.resultList:
            pt1 = (int(result.bbx.xmin * self.c_w), int(result.bbx.ymin * self.c_h))
            pt2 = (int(result.bbx.xmax * self.c_w), int(result.bbx.ymax * self.c_h))
            color = result.gender.getGenderColor() if result.qualifiedFace else ageGenderCore.Gender.indexToColor[-1]
            cv2.rectangle(self.currentImage, pt1, pt2, color, 1)