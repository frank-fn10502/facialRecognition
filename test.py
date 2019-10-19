from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import time
import cv2

'''
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
'''
class GenderRecognitionCore:
    def __init__(self):
        pre_t = time.time()
        self.model = load_model('gender-model-base-mobilenet.h5')
        print("create model : {:.4f}s".format(time.time() - pre_t))

    def predectList(self ,faceImgInfoList ,faceImages):
        colorList = [] #(0, 255, 0) ,(255, 0, 0)
        if len(faceImgInfoList) > 0: 
            '''
            faceImages = [ (img  / 255.) for img in faceImages ]
            faceImages = np.array(faceImages)
            print('after: ' ,faceImages)
            #faceImages = np.expand_dims(faceImages, axis=0)
            #print('before: ' ,faceImages)
            
            pred_list = self.model.predict(faceImages)
            pred_list = np.argmax(pred_list, axis=1)
            print(pred_list)
            for info ,p in zip(faceImgInfoList ,pred_list):
                colorList.append((0, 255, 0) if 0 else (255, 0, 0))
                info.objName = f"{'female' if 0 else 'male'}"
            '''       
            for info ,faceImg in zip(faceImgInfoList ,faceImages):
                pred = self.predect(faceImg)
                info.objName = f"female:{pred[0][0]:.3f} male:{pred[0][1]:.3f}"
                colorList.append((0, 255, 0) if pred[0][0] > pred[0][1] else (255, 0, 0))
        return colorList

    def predect(self ,img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]     
        return  self.model.predict(img_tensor)   #[[??? ????]]


#img_path = r'D:\Desktop\gender_dl\UTKFace_crop\genderData\male\42_0_2_20170109012231207.jpg.chip.jpg'
#new_image = load_image(img_path)
#pred = model.predict(new_image)
#print(pred)
#print("female:{:.3f} male:{:.3f}".format(pred[0][0] ,pred[0][1]))