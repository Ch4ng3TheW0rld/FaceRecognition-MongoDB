#pip install pillow
#pip install opencv-contrib-python
import cv2
import numpy as np
import os 
from pymongo import MongoClient
import datetime
import pprint

###Haar alto consumo de CPU y solo spot frontal. (no cetera en reconocimiento)
###en V3 agregar reconocimiento de ojo y smile, envio de correo cuando es unknown
####issue cuando esta vacio el trainer, muestra error TypeError: 'NoneType' object has no attribute '__getitem__' ,id!=Num en este caso debera de forzar que sea "0"
####este script es para webcam de laptop, y con camara externa??


# Initialize and start realtime video capture
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')

#iniciate id counter
id = 0

font = cv2.FONT_HERSHEY_SIMPLEX


def getProfile(id):
    client=MongoClient('192.168.122.173',27017)
    db=client['FaceRecBase']
    collection=db['credentials']
    profile = collection.find_one({"id":id},{"name":1}) 
    profile = profile["name"]
    return profile

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        ###Haar hace analisis y aqui define el reconocimiento y su id
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        ###garantiza que el valor de id sea numerico
        #try:
        #    val = str(id)
        #    pprint(val)
        #except ValueError:
        #    id = 0 


        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = str(id)
            profile = getProfile(id)
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            profile = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(profile), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()