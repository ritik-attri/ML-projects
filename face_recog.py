import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import time

files=[files for files in os.listdir('imgdataset/') if files.endswith('.npy')]
print(files)
names=[name[:-4] for name in files]
print(names)
face_data=[]
for filename in files:
    data=np.load('imgdataset/'+filename)
    face_data.append(data)
face_data=np.array(face_data)
print(face_data.shape)
face_data=np.concatenate(face_data,axis=0)
print(face_data.shape)

label=np.repeat(names,100,axis=0)
print(label)
print(label.shape)
#label=label.reshape((-1,1)) # earlier = (2000,1)
#print(label.shape)


Student=KNeighborsClassifier()
Student.fit(face_data,label)


face_data=[]

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while True:
    time.sleep(0.01)
    ret,frame=cap.read()
    if not ret:
        continue

    

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda x:x[2]*x[3],reverse=True)
    faces=faces[:1]
    print(faces)
    for face in faces:
        x,y,w,h=face
        face_selection=frame[y:y+h,x:x+w]
        print(face_selection.shape)
        face_selection=cv2.resize(face_selection,(100,100))
        face_selection=np.array(face_selection)
        face_selection=face_selection.reshape(1,-1)
        print(face_selection.shape)
        
        predict=Student.predict(face_selection)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        cv2.putText(frame,str(predict),(x+w,y+h),cv2.FONT_HERSHEY_COMPLEX,0.9,(255,0,0),1)

        

    cv2.imshow("Feed",frame)
    key=cv2.waitKey(1)
    if key & 0xFF==ord('q'):
        break		

cap.release()
cv2.destroyAllWindows()


