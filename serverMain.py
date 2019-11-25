import threading 
import cv2
import time
import datetime

#from server.server import ServerCam
from facialRecognitionProcess import Recognition

#myConfig 需要更改


#serverCam = ServerCam()
faceRecoProcess = Recognition()
faceRecoProcess.initAllData()
faceRecoProcess.startCap()

serverCam = faceRecoProcess.serverCam
while True:
    try:
        faceRecoProcess.prev_time = time.time()

        if faceRecoProcess.ret:
            faceRecoProcess.recognized()
            cv2.imshow("result",cv2.cvtColor(faceRecoProcess.currentImage, cv2.COLOR_RGB2BGR))
            
        else:
            print(f'no frame: {datetime.datetime.now()}' ,end = '\r')

    except  ConnectionResetError as e:
        #cv2.destroyAllWindows()
        print("該客戶機異常！已被強迫斷開連接",e)
        print("正在等待連接")

        serverCam.conn , serverCam.addr = serverCam.s.accept()

