import cv2
import numpy as np
import os
from pymongo import MongoClient
import datetime
import pprint


##******Pendiente con la V3 de retomar la data si existe el ID o no
###ver si se mejora el performance
###almacenar la imagen en mongodb

##Almacena Dato
def inputData(ID,NAME):
    client = MongoClient('192.168.122.173',27017) 
    db=client['FaceRecBase']
    collection=db['credentials']

    post={'id':ID,
     'name':NAME,
     'date':datetime.datetime.utcnow()}

    post_id = collection.insert(post)


detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


###ingresar manualmente el valor
id=raw_input('enter your id:')
name=raw_input('enter your name:')

inputData(id,name)

###inicia Video
sampleNum=0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/"+str(id) +"."+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()