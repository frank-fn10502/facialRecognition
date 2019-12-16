import sys
import time
import os
import numpy as np
import math
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# import some PyQt5 modules
from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

# import my customize window
from GUI.mainGUI import Ui_MainWindow
from GUI.registerDialog import Ui_Dialog

from PIL import Image, ImageDraw, ImageFont

import facialDectionCore
import facialRecognitionCore
from facialRecognitionCore import Identity
from facialRecognitionProcess import Recognition


class RegisterDialog(QDialog):
    def __init__(self):
        # --------------------------設定視窗----------------------------------------------
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.nameLineEdit.setText("testFace")
        self.ui.okButton.clicked.connect(self.setName)

        # --------------------------所需變數----------------------------------------------
        self.name = "myName"

    def setName(self):
        self.name = self.ui.nameLineEdit.text()
        self.accept()

class mainWindows(QMainWindow):
    def __init__(self):
        # --------------------------設定視窗----------------------------------------------
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.timer = QTimer()
        self.timer.timeout.connect(self.detectFaces)
        self.ui.start_stop_button.clicked.connect(self.startAndStopButton)
        self.ui.cutButton.clicked.connect(self.registerButton)
        self.ui.manualCapButton.clicked.connect(self.reCapButton)

        # --------------------------所需變數---------------------------------------------
        self.recognized = True  # True#False
        self.registered = False

        self.faceRecoProcess = Recognition()
        if self.faceRecoProcess.initAllData():  #False 代表 cfg 檔案中有問題
            if self.faceRecoProcess.cap != None:
                if not self.faceRecoProcess.cap.isOpened():
                    self.ui.start_stop_button.setEnabled(False)
                    self.ui.cutButton.setEnabled(False)
    
                else:
                    self.ui.manualCapButton.setEnabled(False)
            else:
                self.ui.manualCapButton.setEnabled(False)
                
        else:
            self.faceRecoProcess = None
            

    def closeEvent(self, event):
        self.timer.stop()
        self.faceRecoProcess.stopAll()
        if self.faceRecoProcess.cap is not None:
            self.faceRecoProcess.stopCap()
            self.faceRecoProcess.cap.release()

    def startAndStopButton(self):
        if not self.timer.isActive():
            self.timer.start(5)
            self.ui.start_stop_button.setText("暫停偵測")
            self.faceRecoProcess.startCap()
        else:
            self.timer.stop()
            self.ui.start_stop_button.setText("開始偵測")
            self.faceRecoProcess.stopCap()

    def registerButton(self):
        if self.registered:
            self.ui.cutButton.setText("註冊")
        else:
            self.ui.cutButton.setText("取消註冊")
            self.faceRecoProcess.initRegData()

        self.registered = not self.registered

    def reCapButton(self):
        self.faceRecoProcess.initCapDevice()
        
        if self.faceRecoProcess.cap.isOpened():
            self.ui.start_stop_button.setEnabled(True)
            self.ui.cutButton.setEnabled(True) 
            self.ui.manualCapButton.setEnabled(False)  

    def cvImgConvertToQImage(self, sourceImg, BGR2RGB=False):
        rgbImage = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB) if BGR2RGB else sourceImg
        height, width, channel = rgbImage.shape
        rgbImage = QImage(rgbImage.data ,width ,height ,channel * width ,QImage.Format_RGB888)

        return rgbImage

    def detectFaces(self):  # timer每次呼叫的function
        self.faceRecoProcess.prev_time = time.time()

        if self.faceRecoProcess.ret:
            if self.registered:  # 註冊
                needName ,faceImg = self.faceRecoProcess.registered()
                if needName:
                    regDialog = RegisterDialog()
                    regDialog.ui.faceImageLabel.setPixmap(QPixmap.fromImage(self.cvImgConvertToQImage(faceImg)))

                    if regDialog.exec_():
                        self.faceRecoProcess.setRegName(regDialog.name)
                    else:
                        self.registered = False
                        self.ui.cutButton.setText("註冊")
                        print('儲存照片失敗')
                
                if self.faceRecoProcess.isRegDone():
                    self.registered = False
                    self.ui.cutButton.setText("註冊")
                
            elif self.recognized:  # 辨識
                self.faceRecoProcess.recognized()

            self.ui.display_label.setPixmap(QPixmap.fromImage(self.cvImgConvertToQImage(self.faceRecoProcess.currentImage)))  #show current frame in img_label
        else:
            print(f'no frame: {datetime.datetime.now()}' ,end = '\r')


def main():
    app = QApplication(sys.argv)
    windows = mainWindows()
    if windows.faceRecoProcess is not None:
        windows.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()