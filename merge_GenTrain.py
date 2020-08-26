import cv2,os
import numpy as np
from pymongo import MongoClient
import datetime
import pprint
from PIL import Image

##******Pendiente con la V3 de retomar la data si existe el ID o no
###ver si se mejora el performance
###almacenar la imagen en mongodb

##Almacena Dato
def inputData(ID,NAME):
    client = MongoClient('192.168.122.173',27017) 
    db=client['FaceRecBase']
    collection=db['credentials']

    post={"id":ID,"name":NAME,"date":datetime.datetime.utcnow()}
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


recognizer = cv2.face.LBPHFaceRecognizer_create() #cv2.createLBPHFaceRecognizer()

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:

        # Updates in Code
        # ignore if the file does not have jpg extension :
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')