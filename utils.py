import os
import time
import threading
import json

from enum import Enum

class Dot:
    def __init__(self ,x ,y):
        self.x = x
        self.y = y


class Square:
    def __init__(self ,w ,h):
        self.w = w
        self.h = h

    def getTuple(self):
        return (self.w ,self.h)



class FacialFeature:
    def __init__(self ,qualifiedFace = True):    
        self.qualifiedFace = qualifiedFace

        self.bbx = None     #####
        self.identity = None
        self.gender = None  #####
        self.age = None     #####
        self.emotion = None

    def __str__(self):
        return f"{str(self.identity)} "


class CircularQueue:
    def __init__(self ,maxsize = 24):
        self.maxSize = maxsize  
        self.count = 0      
        self.front = 0
        self.rear  = 0 
        self.queue = [''] * maxsize    

    def push(self ,data):
        self.queue[self.rear] = data
        self.rear = (self.rear + 1) % self.maxSize
        if self.rear == self.front:
            self.front = (self.front + 1) % self.maxSize

        self.count = 1 + self.count if self.count < self.maxSize else self.count

    def size(self):
        return len(self.queue)

    def len(self):
        return self.count

    def isFull(self):
        return self.count == len(self.queue)

    def getList(self):
        l = []
        counter = 0
        i = self.front
        while counter < self.len():
            l.append(self.queue[i])

            i = (i + 1) % self.size() 
            counter += 1

        return l


class OutputHandler:
    def __init__(self ,fileName = "./other/outputData/data.txt",jsonFilePath = './other/outputData/detectionData.json'):
        self.write = True
        self.textList = []
        self.fileName = fileName
        self.jsonFilePath = jsonFilePath

        os.makedirs(os.path.dirname(self.fileName), exist_ok=True)
        os.makedirs(os.path.dirname(self.jsonFilePath), exist_ok=True)

    def start(self):       
        threading.Thread(target=self.__writeData).start()

    def addResult(self ,text):
        self.textList.append(text)

    def __writeData(self):
        with open(self.fileName , "a") as myfile:
            while self.write:
                if len(self.textList) > 0:
                    myfile.write(f"{self.textList.pop(0)}\n")
                time.sleep(0.05)

    def createJsonData(self ,jsonData ,needPrint = False ,writeJson = False):
        myData = json.dumps(jsonData ,ensure_ascii=False)

        if needPrint:
            print(myData)

        if writeJson:
            with open(self.jsonFilePath ,'w' ,encoding='UTF-8') as jf:
                jf.write(myData)      

        return myData    


class RecType(Enum):
    SERVER = 'server'
    CLIENT = 'client'
    LOCALHOST = 'localhost'