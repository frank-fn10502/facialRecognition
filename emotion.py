from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2



class EmotionRec:
    def __init__(self):
        self.model  = None#self.__initModel()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


    def initModel(self):
        json_file = open('./emotion/fer.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./emotion/fer.h5")

        print("Loaded model from disk")
        self.model = loaded_model
        #return loaded_model

    def predict(self ,faceImgInfoList ,images):
        imgList = []
        if len(faceImgInfoList) > 0:
            for img in images:
                img_gray = cv2.cvtColor(img ,cv2.COLOR_RGB2GRAY)
                img_temp = cv2.resize(img_gray, (48 ,48), interpolation=cv2.INTER_CUBIC)
                img_temp = np.expand_dims(img_temp, -1)

                imgList.append(img_temp)
            imgList = np.array(imgList)

            preds= self.model.predict(imgList)
            result = []
            for i in range( len(preds) ):
                result.append( preds[i][:] )
            
            for info ,emotion in zip(faceImgInfoList ,result):
                e_index = np.argmax(np.array(emotion) ,axis=0)
                info.objName = f"{info.objName} ,{self.emotion_labels[e_index]}"