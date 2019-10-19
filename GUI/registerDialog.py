# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'registerDialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 330)
        self.faceImageLabel = QtWidgets.QLabel(Dialog)
        self.faceImageLabel.setGeometry(QtCore.QRect(70, 0, 231, 250))
        self.faceImageLabel.setText("")
        self.faceImageLabel.setPixmap(QtGui.QPixmap("GUI_Img/illuminati.jpg"))
        self.faceImageLabel.setScaledContents(False)
        self.faceImageLabel.setObjectName("faceImageLabel")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 250, 111, 31))
        self.label_2.setObjectName("label_2")
        self.nameLineEdit = QtWidgets.QLineEdit(Dialog)
        self.nameLineEdit.setGeometry(QtCore.QRect(10, 280, 251, 31))
        self.nameLineEdit.setObjectName("nameLineEdit")
        self.okButton = QtWidgets.QPushButton(Dialog)
        self.okButton.setGeometry(QtCore.QRect(290, 270, 101, 41))
        self.okButton.setObjectName("okButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "註冊"))
        self.label_2.setText(_translate("Dialog", "輸入註冊姓名:"))
        self.okButton.setText(_translate("Dialog", "確認"))
