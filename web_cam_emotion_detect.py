import cv2
from fastbook import * 
from fastai.vision.widgets import *
import pathlib 
import matplotlib.pyplot as plt 
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
fp = Path('D:/project/face_emo/emo7.pkl')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
learn_inf = load_learner(fp)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    j =1
    for (x , y, w, h) in faces:
        frame = cv2.rectangle(frame , (x, y) , (x+w , y+h) , (255 , 255,0) , 2)
        frame_gray = cv2.rectangle(frame , (x, y) , (x+w , y+h) , (255 , 255,0) , 2)
        face = frame_gray[y:y+h , x : x+w]
    pred , pred_inx , probs  = learn_inf.predict(gray)
    cv2.putText(frame,
                    pred,
                    (50,50), 
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,0),2)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


