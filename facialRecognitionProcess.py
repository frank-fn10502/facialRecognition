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


import darknet

from PIL import Image, ImageDraw, ImageFont

import facialDectionCore
import facialRecognitionCore
from facialRecognitionCore import Identity
#import test
import age_gender
import emotion

class CircularQueue:
    def __init__(self ,maxsize = 24):
        self.maxSize = maxsize        
        self.front = 0
        self.rear  = 0 
        self.queue = [''] * maxsize    

    def push(self ,data):
          self.queue[self.rear] = data
          self.rear = (self.rear + 1) % self.maxSize
          if self.rear == self.front:
              self.front = (self.front + 1) % self.maxSize

    def size(self):
        return len(self.queue)

class AccData:  #Accumulated recognized facial data
    def __init__(self ,faceImageInfo ,name = ''):
        #-------------------變數----------------------------
        self.treshDisplay = 0.6
        self.dataList = CircularQueue()     #[(name1 ,info1) ,(name2 ,info2) , ...]

        self.num_counter = 0
        self.counter  = 5         #設定存活的次數(如果都偵測不到)
        self.treshIOU = 0.8
        self.lastFaceImageInfo = None
        self.mostPossibleName  = ''
        self.mostPossibleObjName  = ''
        #-------------------設定----------------------------       
        self.addInfo(faceImageInfo ,name)

    def addInfo(self ,faceImgInfo ,name = ''):
        self.counter = 5
        self.lastFaceImageInfo = faceImgInfo
        
        self.dataList.push( (name ,faceImgInfo) )
        if self.num_counter < self.dataList.maxSize:
            self.num_counter += 1
        
    def isThisAccData(self ,faceImgInfo):
        #isPass ,IOU = self.__calIOU(faceImgInfo)
        #if isPass:
        #    print("iou:{0}".format(IOU) )      

        return self.__calIOU(faceImgInfo) #isPass 

    def adjustBBX(self ,faceImgInfo ,mainImage):
        w = 0
        h = 0
        count = 0
        for data in self.dataList.queue:
            #with data as (name ,info):
            if data != '' and data[1] != '':
                w += data[1].w
                h += data[1].h
                count += 1
                       
        w += faceImgInfo.w 
        h += faceImgInfo.h 
        count += 1

        w /= count
        h /= count
        faceImgInfo.calBBXShape( [faceImgInfo.objName ,faceImgInfo.confScore , [faceImgInfo.yolo_x ,faceImgInfo.yolo_y ,w ,h]] 
                               , mainImage)        

    def isStillAlive(self):
        self.counter -= 1
        return True if self.counter >= 0 else False

    def calResult(self):
        tempDict_name = {}
        gender_temp_list = [0 ,0] #male ,female
        e_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        e_list = [0 ,0 ,0 ,0 ,0 ,0 ,0]
        age_temp_avg = 0

        i = self.dataList.front
        counter = 0
        defect  = 0
        while counter < self.num_counter:
            list_data = self.dataList.queue[i][1].objName.split(',')
            if len(list_data) > 1:
                gender ,age ,emotion = list_data
                gender = gender.strip()
                age = int(age)
                if gender == 'male':
                    gender_temp_list[0] += 1
                else:
                    gender_temp_list[1] += 1

                for index ,em in enumerate(e_labels) :
                    if em == emotion:
                        e_list[index] += 1

                age_temp_avg += age
            else:
                defect += 1
            #with self.dataList.queue[i] as (name ,info):
            if tempDict_name. __contains__(self.dataList.queue[i][0]): 
                tempDict_name[self.dataList.queue[i][0]] += 1
            else:
                tempDict_name[self.dataList.queue[i][0]] = 1
            i = (i + 1) % self.dataList.size() 
            counter += 1

        emotion = e_labels[np.argmax(e_list,axis=0)] 
        if counter - defect != 0:
            age_temp_avg /= (counter - defect)
        else:
            age_temp_avg = 18

        if gender_temp_list[0] > gender_temp_list[1]:
            self.mostPossibleObjName = f"male ,{round(float(age_temp_avg))} ,{ emotion }"
        elif gender_temp_list[1] > gender_temp_list[0]: 
            self.mostPossibleObjName = f"female ,{round(float(age_temp_avg))} ,{ emotion }"
        else:
            self.mostPossibleObjName = f"eq ,{round(float(age_temp_avg))} ,{ emotion }"

        currentMax = 0 
        self.mostPossibleName = 'noMan'   
        for accName ,count in tempDict_name.items():
            if count >= currentMax and count >= math.ceil(self.treshDisplay * self.num_counter):
                currentMax = count
                self.mostPossibleName = accName   

        print(f"name:{tempDict_name}  ,gender:{gender_temp_list}  ,age:{age_temp_avg :.2f}  ,emotion:{emotion}  ,num:{self.num_counter}")

    def __calIOU(self ,anotherFaceImageInfo):
        #如果有交集才計算
        if(abs(self.lastFaceImageInfo.centroid[0] - anotherFaceImageInfo.centroid[0]) < (self.lastFaceImageInfo.w + anotherFaceImageInfo.w) / 2 and
           abs(self.lastFaceImageInfo.centroid[1] - anotherFaceImageInfo.centroid[1]) < (self.lastFaceImageInfo.h + anotherFaceImageInfo.h) / 2):
           inter_x1 = max(self.lastFaceImageInfo.xmin ,anotherFaceImageInfo.xmin)
           inter_y1 = max(self.lastFaceImageInfo.ymin ,anotherFaceImageInfo.ymin)
           inter_x2 = min(self.lastFaceImageInfo.xmax ,anotherFaceImageInfo.xmax)
           inter_y2 = min(self.lastFaceImageInfo.ymax ,anotherFaceImageInfo.ymax)
           inter_square = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
           union_square = self.lastFaceImageInfo.w * self.lastFaceImageInfo.h + anotherFaceImageInfo.w * anotherFaceImageInfo.h - inter_square

           IOU = inter_square / union_square * 1.0

           return IOU >= self.treshIOU ,IOU
        else:
            return False ,0

class OutputHandler:
    def __init__(self):
        self.write = True
        self.textList = []
        self.fileName = "./outputData/data.txt"

    def start(self):
        os.makedirs(os.path.dirname(self.fileName), exist_ok=True)

        threading.Thread(target=self.__writeData).start()
        print("outputDone!!!")

    def addResult(self ,text):
        self.textList.append(text)

    def __writeData(self):
        with open(self.fileName , "a") as myfile:
            while self.write:
                if len(self.textList) > 0:
                    myfile.write(f"{self.textList.pop(0)}\n")
                time.sleep(0.05)

class Recognition:
    def __init__(self):        
        # --------------------------設定人臉偵測演算法------------------------------------
        self.facialDetection = facialDectionCore.FacialDectionCore()  # 讀取並設定YOLO
        # ------------------------------------設定人臉辨識演算法--------------------------
        self.facialRecognition = facialRecognitionCore.FacialRecognitionCore()  

        #------------初始化mobilenet模型和資料-gender---------------------------        
        #self.genderRecognition = test.GenderRecognitionCore()
        self.genderAgeRec = age_gender.GenderAgeRec()

        #------------初始化模型和資料-emotion----------------------------------       
        self.emotionRec = emotion.EmotionRec()

        #---------------OPENCV---------------------------------------------
        self.capDevice = 0 #'http://admin:123456@140.126.20.95/video.cgi'#0  # "./data/YouTube_Rewind_2016.mp4"
        self.deviceW = 1920
        self.deviceH = 1080
        self.cap = None
        self.sock = socket.socket()#遠端攝影機
        

        #--------------參數----------------------------------------------
        self.jsonFilePath = 'D:/Desktop/detectionData.json'
        self.writeJson    = False#True
        self.maxFaceCount = 10
        self.saveInterval = 0.5
        self.cfgFile = './cfg/myConfig.cfg'
        
        self.colorList = []
        self.possiblePassInfoList = []
        #----------------------------------------------------------------
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
        #warnings.filterwarnings("ignore")
        #warnings.simplefilter(action='ignore', category=FutureWarning)
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #----------------讀取cfg資料----------------------------------
        if not self.__initCfgData():
            return False
        
        pre_t = time.time()
        #----------------初始化yolo模型-------------------------------
        t1 = threading.Thread( target=self.facialDetection.initYOLO )
        #self.facialDetection.initYOLO()

        #------------初始化facenet模型和資料---------------------------        
        #t2 =  threading.Thread( target=self.facialRecognition.initFacenet ,args=("./module/20180402-114759.pb" ,) )
        #t3 =  threading.Thread( target=self.facialRecognition.initFolderImage )
        t1.start()
        #t2.start()
        #t3.start()
        self.facialRecognition.initFacenet("./module/20180402-114759.pb")# 初始化facenet模型和資料 20180402-114759.pb 20180408-102900.pb       
        self.facialRecognition.initFolderImage()

        #------------初始gender_age-----------------------------------  
        self.genderAgeRec.initModel()

        #------------初始gender_age-----------------------------------  
        self.emotionRec.initModel()

        #------------初始攝影機參數------------------------------------    
        self.__initCapDevice()

        t1.join()
        #t2.join()
        #t3.join()

        print(f'perpare all cost:{time.time() - pre_t:.3f}')
        return True

    def startCap(self):
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
        faceImgInfoList ,resultList = self.__preProcess()
        regFaceImg  = None
        regFaceInfo = None
        for info ,result in zip(faceImgInfoList ,resultList):
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

                for info in faceImgInfoList:
                    self.displayTextList.append(["完成:{0} %".format(self.faceitem * 10), (info.xmin, info.ymin), 20])

        if displayInfo:
            cv2.rectangle(self.currentImage, self.regAreaPt1, self.regAreaPt2,(255, 0, 0), 3)  #劃出註冊的有效區域 
            colorList = [ ( 0 ,0 ,255) for i in range(len(faceImgInfoList))]
            self.__drawOnImg(faceImgInfoList ,colorList)
            #self.__drawOnImg(faceImgInfoList)

        return self.faceitem == 1 and self.needName ,regFaceImg

    def recognized(self  ,displayInfo = True):
        pre_t = time.time()
        faceImgInfoList ,resultList = self.__preProcess()
        yolo_time = time.time() - pre_t  

        faceImages = []
        faceImage_ori_list = []
        for info in faceImgInfoList:
            # 裁剪座標為[y0:y1 ,x0:x1]
            croppedImage = self.resizeImg[info.ymin:info.ymax ,info.xmin:info.xmax]
            croppedImage = cv2.cvtColor(croppedImage ,cv2.COLOR_BGR2RGB)
            faceImages.append(croppedImage)

            pt1 = (int(info.xmin * self.c_w), int(info.ymin * self.c_h))
            pt2 = (int(info.xmax * self.c_w), int(info.ymax * self.c_h))
            croppedImage = self.currentImage[pt1[1]:pt2[1] ,pt1[0]:pt2[0]]
            croppedImage = cv2.cvtColor(croppedImage ,cv2.COLOR_BGR2RGB)

            h ,w ,c = croppedImage.shape
            border_num = h - w if h > w else w - h
            border_num = math.ceil(border_num / 2.0)
            direction = 1 if w > h else 0 #1:上和下 0:左和右
            img_patch = None
            if direction == 1:
                img_patch = cv2.copyMakeBorder(croppedImage, border_num, border_num, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                img_patch = cv2.copyMakeBorder(croppedImage,  0, 0, border_num, border_num, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            cv2.imwrite('./temp/output.jpg', img_patch)    
            faceImage_ori_list.append(img_patch)
        
        #t2 =  threading.Thread(target = self.__recID ,args=(faceImages ,))
        #t2.start()
        
        #####
        colorList = []
        pre_t = time.time()
        #colorList = self.genderRecognition.predectList(faceImgInfoList ,faceImages)
        colorList = self.genderAgeRec.display(faceImgInfoList ,faceImage_ori_list)
        g_time = time.time() - pre_t    
        
        #####
        
        pre_t = time.time() 
        possiblePassInfoList = self.facialRecognition.compareFace(faceImages)  # 讓"所有裁切下來的人臉"和"資料庫的人臉"做比對 (回傳 passInfo) 
        id_time = time.time() - pre_t
        
        pre_t = time.time()
        self.emotionRec.predict(faceImgInfoList ,faceImage_ori_list)
        e_time = time.time() - pre_t    
        #t1 = threading.Thread(target = self.__recGender_Age ,args=(faceImgInfoList ,faceImage_ori_list ,))   
        #t1.start()
        #t1.join()
        #t2.join()

        #colorList = self.colorList
        #possiblePassInfoList = self.possiblePassInfoList
        

        if len(possiblePassInfoList) > 0:
            for index ,(possiblePassInfo, info ,result) in enumerate( zip(possiblePassInfoList, faceImgInfoList ,resultList) ):
                if result is not None:
                    print(f"{index}. " ,end='')
                    result.calResult()
                    mostPossibleObjName = result.mostPossibleObjName.strip('()')
                    #print(mostPossibleObjName)
                    #print(mostPossibleObjName.split(' ,')[0])

                    if mostPossibleObjName.split(' ,')[0] != 'eq':
                        colorList[index] = (0, 255, 0) if mostPossibleObjName.split(' ,')[0]  == 'male' else (255, 0, 0)

                    if displayInfo:
                        self.displayTextList.append(["{0} [{1}][{2}][{3}]".format(result.mostPossibleName 
                                                                        ,str(round(possiblePassInfo.passAvgDist, 4)) 
                                                                         if possiblePassInfo.name == result.mostPossibleName and possiblePassInfo.name != 'noMan' else 
                                                                         'unKnow' 
                                                                        ,round(info.confScore * 100, 2)
                                                                        ,mostPossibleObjName if mostPossibleObjName.split(' ,')[0] != 'eq' else info.objName)
                                                                        ,(info.xmin, info.ymin)
                                                                        ,20])

                    result.addInfo(info ,possiblePassInfo.name)
                else:
                    self.accDataList.append(AccData(info ,possiblePassInfo.name))
                    if displayInfo:
                        self.displayTextList.append(["{0} [{1}][{2}][{3}]".format(possiblePassInfo.name 
                                                                    ,str(round(possiblePassInfo.passAvgDist, 4))  
                                                                    ,round(info.confScore * 100, 2)
                                                                    ,info.objName) 
                                                    ,(info.xmin, info.ymin)
                                                    ,20])  
        else:
            if displayInfo:
                for info in faceImgInfoList:
                    self.displayTextList.append(["{0} [{1}][{2}][{3}]".format("noMan", "unKnow", round(info.confScore * 100, 2),info.objName) 
                                        ,(info.xmin, info.ymin)
                                        ,20])   
        if displayInfo:   
            self.__drawOnImg(faceImgInfoList ,colorList)

        print(f"reco gender_age: {g_time:.2f}s | reco id:{id_time:.2f}s | yolo: {yolo_time:.2f} | emotion: {e_time:.2f} | total time: {id_time + g_time + yolo_time + e_time:.4f}") 
        print("=" * 40 ,"\n")  

        self.postAnswer()
        return self.__createJsonFile(faceImgInfoList)
 
    def stopAll(self):
        self.outputHandler.write = False
    
    def postAnswer(self):
        my_data = {'msg' : 'f3'}
        for displayText in self.displayTextList:
            data = displayText[0].split(' [')[0]
            if data != 'noMan':
                my_data['msg'] = 'n3'
                break
        
        r = requests.post(' http://localhost/IOT/V2/control/pubControl.php', data = my_data)
        

    #-----------------------------private func----------------------------------------------
    def __recGender_Age(self ,faceImgInfoList ,faceImage_ori_list):
        print(f'start thread')
        print(f'faceImgInfoList: {len(faceImgInfoList)}')
        print(f'faceImgInfoList: {faceImage_ori_list}')
        pre_t = time.time()
        #colorList = self.genderRecognition.predectList(faceImgInfoList ,faceImages)
        self.colorList = self.genderAgeRec.display(faceImgInfoList ,faceImage_ori_list)
        g_time = time.time() - pre_t    
        print(f'Done g_a cost:{g_time}')

    def __recID(self ,faceImages):
        pre_t = time.time()
        self.possiblePassInfoList = self.facialRecognition.compareFace(faceImages)  # 讓"所有裁切下來的人臉"和"資料庫的人臉"做比對 (回傳 passInfo) 
        id_time = time.time() - pre_t
        print(f'Done g_a cost:{id_time}')

    def __drawOnImg(self ,faceImgInfoList ,colorList):
        if colorList == []:
            colorList.append( (0, 255, 0) )
        # ---------------------劃出文字---------------------------------------------------
        self.__displayText(self.displayTextList)   #[str ,pos ,fontSize]

        #----------------------劃出bbx---------------------------------------------------- 
        self. __drawBBX(faceImgInfoList ,colorList )          

        # ---------------------顯示圖片---------------------------------------------------
        fps = 1 / (time.time() - self.prev_time)
        processTime = time.time() - self.prev_time
        cv2.putText(self.currentImage ,"FPS:{0:.2f}  time:{1:.5f}  faceNum:{2}".format(
                    fps, processTime, len(faceImgInfoList) ),
                    (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 0, 0], 1)

        self.outputHandler.addResult(f"{datetime.datetime.now()} {processTime:.4f} {len(faceImgInfoList)}")
        print( f"height:{self.currentImage.shape[0]} ,width:{self.currentImage.shape[1]}" \
               f" fps:{fps:.2f} time:{processTime:.5f} faceNum:{len(faceImgInfoList)}")

    def __createJsonFile(self ,faceImgInfoList):
        if faceImgInfoList is not None:
            dataList = []
            for displayText ,info in zip(self.displayTextList ,faceImgInfoList):
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

            self.__insureHasFolder(self.jsonFilePath)
            print(f'jsonFilePath: {self.jsonFilePath}')
            self.__insureHasFolder(self.facialRecognition.fileRootPath)

        except FileNotFoundError:
            print('你的cfg檔案好像不見了!!!')

        except Exception as e:
            print(e)
            return False

        return True

    def __insureHasFolder(self ,path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

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
                self.capHeight = int(sp[0])#int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.capWidth  = int(sp[1])#int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.regAreaW = int(self.capWidth/4)  if self.regAreaW == -1 else int((self.capWidth  - self.regAreaW)/2)
                self.regAreaH = int(self.capHeight/4) if self.regAreaH == -1 else int((self.capHeight - self.regAreaH)/2)
                print(f'regAreaW: {self.regAreaW} ,regAreaH:{self.regAreaH}')

                self.regAreaPt1 = (self.regAreaW, self.regAreaH )
                self.regAreaPt2 = (self.capWidth - self.regAreaW ,self.capHeight - self.regAreaH)
                self.ret = True
                self.frame = decimg
                self.c_w = sp[1]/ darknet.network_width(self.facialDetection.netMain)#self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  / darknet.network_width(self.facialDetection.netMain)
                self.c_h = sp[0]/ darknet.network_height(self.facialDetection.netMain)#self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / darknet.network_height(self.facialDetection.netMain) 
  

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

    def __getResultList(self, faceImgInfoList): #用累計的datalist推測出結果
        resultList = []                         #對應到相對的 累計list(用這list推測出結果)
        for index ,info in enumerate(faceImgInfoList):
            
            hasAcc = False
            for accData in self.accDataList:
                isPass ,IOU = accData.isThisAccData(info)
                if isPass:
                    accData.adjustBBX(info ,self.resizeImg)                    
                    resultList.append(accData)
                    hasAcc = True
                    break
            
            print(f"{index}. iou: {IOU if hasAcc else 'noMatch'}")
            if not hasAcc:
                resultList.append(None)
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

        self.resizeImg ,faceImgInfoList = self.facialDetection.captureData(self.currentImage)  #currentImage的畫面 ,從yolo網路中取得bbx的參數(存成list)
        resultList = self.__getResultList(faceImgInfoList)                                     #從累計的list中取得結果

        self.currentImage = cv2.cvtColor(self.currentImage, cv2.COLOR_BGR2RGB)
        return faceImgInfoList ,resultList

    def __displayText(self ,displayTextList):
        fontSize = 0
        color    = (255, 0, 0)
        imgPIL   = Image.fromarray(self.currentImage)
        draw     = ImageDraw.Draw(imgPIL)

        for textInfo in displayTextList:
            if fontSize != textInfo[2]:
                fontSize = textInfo[2]
                font = ImageFont.truetype('simsun.ttc', fontSize)
            
            draw.text((textInfo[1][0] * self.c_w ,textInfo[1][1] * self.c_h  - 20), textInfo[0], font=font, fill=color)  #(pos ,str ,font ,color)          
        self.currentImage = np.asarray(imgPIL)

    def __drawBBX(self ,faceImgInfoList ,colorList):
        for info ,color in zip(faceImgInfoList ,colorList):
            pt1 = (int(info.xmin * self.c_w), int(info.ymin * self.c_h))
            pt2 = (int(info.xmax * self.c_w), int(info.ymax * self.c_h))
            cv2.rectangle(self.currentImage, pt1, pt2, color,1)