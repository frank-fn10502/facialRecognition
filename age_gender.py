import os
import time
import cv2
import numpy as np

from keras.models import load_model
from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

class GenderAgeRec():
    def __init__(self):
        self.index_to_gender = {0: 'male', 1: 'female'}
        

    def initModel(self):
        self.model = load_model('weights/gender-age-mobilenetv1_frank_ver3_3_50_v1.h5' ,compile=False)
        #gender-age-mobilenetv2_frank_ver3_5 
        #gender-age-mobilenetv1_frank_ver3_3_50_v1\


    def predict(self ,images):
        images_resize_list = []
        for img in images:
            img_temp = cv2.resize(img, (224 ,224))
            img_temp = preprocess_input(img_temp)   
            images_resize_list.append(img_temp)
        
        images_np = np.array(images_resize_list)  
        return self.model.predict(images_np) 

    def display(self ,faceImgInfoList ,faceImages):
        colorList = []
        if len(faceImgInfoList) > 0:
            preds = self.predict(faceImages)
            result = []
            #print(preds)
            for i in range( len(preds[0]) ):
                result.append([ preds[0][i,0] ,preds[1][i,:] ])

            for info ,(age ,gender) in zip(faceImgInfoList ,result):
                g_index = np.argmax(gender,axis=0)
                info.objName = f"{self.index_to_gender[g_index]} ,{round( float(age) )}"
                colorList.append( (0, 255, 0) if g_index == 0 else (255, 0, 0) )

        return colorList    #(0, 255, 0) ,(255, 0, 0)