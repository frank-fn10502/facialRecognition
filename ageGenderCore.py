import os
import time
import cv2
import numpy as np

from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
#from keras.applications.mobilenetv2 import preprocess_input
#from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from utils import FacialFeature

class Gender:
    indexToColor  = {-1 : (207 ,207 ,207) , 0: (0, 255, 0), 1: (255, 0, 0)} 
    indexToGender = {-1 : '' ,0: 'male', 1: 'female'}
    def __init__(self ,index = -1):
        self.genderIndex = index

    def getGenderName(self):
        return Gender.indexToGender[self.genderIndex]

    def getGenderColor(self):
        return Gender.indexToColor[self.genderIndex]

class GenderAgeRec():
    def __init__(self):
        pass
        #self.index_to_gender = {0: 'male', 1: 'female'}
        

    def initModel(self):
        #self.model = load_model('weights/gender-age-Inceptionv3-20__11_20.h5' ,compile=False)
        self.model = load_model('weights/gender-age-mobilenetv1_frank_ver3_3_50_v1.h5' ,compile=False)
        #gender-age-mobilenetv2_frank_ver3_5 
        #gender-age-mobilenetv1_frank_ver3_3_50_v1\

    def __preProcess(self ,images):
        images_resize_list = []
        for img in images:
            img_temp = cv2.resize(img, (224 ,224))
            img_temp = preprocess_input(img_temp)   
            images_resize_list.append(img_temp)    

        return images_resize_list

    def predict(self ,facialFeatureList ,images):
        if len(images) > 0:
            resultList = []
            
            images = self.__preProcess(images)
            images_np = np.array(images)  
            preds = self.model.predict(images_np)

            for i in range( len(preds[0]) ):
                resultList.append([ preds[0][i,0] ,preds[1][i,:] ])   

            q_facialFeatureList = []
            for f in facialFeatureList:
                if f.qualifiedFace:
                    q_facialFeatureList.append(f)

            for f ,(age ,gender) in zip(q_facialFeatureList ,resultList):
                g_index = np.argmax(gender,axis=0)
                f.gender = Gender(g_index)
                f.age = age

    '''
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
    '''