import cv2
import numpy as np
import time 

name=input("Enter your mood:-")
num=int(input("Enter the number of photos:-"))
face_data=[]


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while True and num:
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
        print(face_selection.shape)
        face_data.append(face_selection)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        num-=1

        

    cv2.imshow("Feed",frame)
    key=cv2.waitKey(1)
    if key & 0xFF==ord('q'):
        break		
print(face_selection.shape)
print(len(face_data))
face_data=np.array(face_data)
print(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)
print(face_data)
np.save(("imgdataset/"+name),face_data)
cap.release()
cv2.destroyAllWindows()
