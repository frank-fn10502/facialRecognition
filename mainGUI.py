# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainGUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.resize(832, 832)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.display_label = QtWidgets.QLabel(self.centralwidget)
        self.display_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.display_label.sizePolicy().hasHeightForWidth())
        self.display_label.setSizePolicy(sizePolicy)
        self.display_label.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.display_label.setAutoFillBackground(False)
        self.display_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.display_label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.display_label.setText("")
        self.display_label.setPixmap(QtGui.QPixmap("GUI_Img/illuminati.jpg"))
        self.display_label.setScaledContents(True)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)
        self.display_label.setWordWrap(False)
        self.display_label.setObjectName("display_label")
        self.horizontalLayout.addWidget(self.display_label)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.start_stop_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_stop_button.sizePolicy().hasHeightForWidth())
        self.start_stop_button.setSizePolicy(sizePolicy)
        self.start_stop_button.setObjectName("start_stop_button")
        self.horizontalLayout_3.addWidget(self.start_stop_button)
        self.cutButton = QtWidgets.QPushButton(self.centralwidget)
        self.cutButton.setEnabled(True)
        self.cutButton.setObjectName("cutButton")
        self.horizontalLayout_3.addWidget(self.cutButton)
        self.manualCapButton = QtWidgets.QPushButton(self.centralwidget)
        self.manualCapButton.setEnabled(True)
        self.manualCapButton.setObjectName("manualCapButton")
        self.horizontalLayout_3.addWidget(self.manualCapButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人臉辨識"))
        self.start_stop_button.setText(_translate("MainWindow", "開始偵測"))
        self.cutButton.setText(_translate("MainWindow", "註冊"))
        self.manualCapButton.setText(_translate("MainWindow", "重設攝像頭"))


