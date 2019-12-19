import cv2

import dlib

import numpy as np

import math

import os

class Preprocessing:
    def __init__(self):
        self.PREDICTOR_PATH = 'weights/shape_predictor_5_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    def landmarks_RotImg(self,im):
#        #rects = [[0,0,im.shape[1],im.shape[0]]]
#        #rects = [pt1[0],pt1[1],pt2[0]-pt1[0],pt2[1]-pt1[1]]
#        pt1 = list(pt1)
#        pt2 = list(pt2)
#        
#        zoon = [40,40,40,40]
#        
#        if pt1[0] >= zoon[0]:
#             pt1[0] = pt1[0] - zoon[0]
#        else:
#            pt1[0] = 0
#            zoon[0] = pt1[0] - 0
#            
#        if pt1[1] >= zoon[1]:
#             pt1[1] = pt1[1] - zoon[1]
#        else:
#            pt1[1] = 0
#            zoon[1] = pt1[1] - 0
#
#        if pt2[0] + zoon[2] <= im.shape[1]:
#             pt2[0] = pt2[0] + zoon[2]
#        else:
#            pt2[0] = im.shape[1]
#            zoon[2] = im.shape[1]-pt2[0]
#
#        if pt2[1] + zoon[3] <= im.shape[0]:
#             pt2[1] = pt2[1] + zoon[3]
#        else:
#            pt2[1] = im.shape[0]
#            zoon[3] = im.shape[0]-pt2[1]
#        
#        im = im[pt1[1]:pt2[1] ,pt1[0]:pt2[0]]
        
        [x,y,w,h] = [0,0,im.shape[1],im.shape[0]]
        shape = None
#        for [x,y,w,h] in rects:
        rect=dlib.rectangle(x,y,x+w,y+h)
        shape = self.predictor(im, rect)
            #dimface = numpy.matrix([[p.x, p.y] for p in shape.parts()])
        
        order = [0,2]
        eye_center =( (shape.part(order[1]).x + shape.part(order[0]).x) * 1./2, # 计算放兩眼的中心坐标
                         (shape.part(order[1]).y + shape.part(order[0]).y) * 1./2) 
        dx = (shape.part(order[0]).x - shape.part(order[1]).x) # note: right - right
        dy = (shape.part(order[0]).y - shape.part(order[1]).y)
    
        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算射矩阵
        RotImg = cv2.warpAffine(im, RotateMatrix, (im.shape[1], im.shape[0]) ,borderMode=cv2.BORDER_REPLICATE) # 進行放這變換(旋轉)
        
        return RotImg
        
#        return RotImg[zoon[0]:im.shape[0]-zoon[2],zoon[1]:im.shape[1]-zoon[3]]

    def annotate_landmarks(self,im, landmarks):#繪製特徵點
    
        im = im.copy()
    
        for idx, point in enumerate(landmarks):
    
            pos = (point[0, 0], point[0, 1])
    
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        
        return im
    
    def brightness(self, img ):
        cols, rows ,a= img.shape
        brightness = np.sum(img) / (255 * cols * rows)
        return brightness
    
    def hisEqulColor(self,img): #直方距離放大
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        equ = cv2.equalizeHist(channels[0], channels[0])
        equ = cv2.merge(channels, ycrcb)
        equ = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        return equ
    
    def lightProcessing(self,img):
        minimum_brightness = 1.5 #閥值
        brightness = self.brightness(img)
        ratio  = brightness/minimum_brightness
        r = cv2.convertScaleAbs(img, alpha = 1 / ratio , beta = 0)
        return r


#p = Preprocessing()
#input_path = r"D:\UTKFace_data\UTKFace_ready_valid" #資料夾目錄
#output_path = r"D:\UTKFace_data\UTKFace_ready_valid_Light_processing\\"
#files= os.listdir(input_path) #得到資料夾下的所有檔名稱
#s = []
#for file in files: #遍歷資料夾
#    if not os.path.isdir(file): #判斷是否是資料夾,不是資料夾才打開
#        f = cv2.imread(input_path+"/"+file)
#        print(file)
#        img = p.lightProcessing(f)
#        cv2.imwrite(output_path+file, img)

