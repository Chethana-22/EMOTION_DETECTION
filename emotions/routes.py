import secrets,os,sys,glob,flask
import tensorflow as tf
from playsound import playsound
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import pandas as pd
import tkinter
from tkinter import *
from flask import render_template,url_for,flash,redirect,abort
from emotions import app,db
from flask_login import login_user,logout_user,login_required
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2 as cv
import numpy as np
import sounddevice as sd

from keras.layers import *
from keras.models import Sequential

  
mp = {'happy':'https://manybooks.net/categories','sad':'https://www.youtube.com/watch?v=F9wbogYwTVM'}

face_classifier = cv.CascadeClassifier('C:\\Users\\cheth\\Desktop\\EMOTION_DETECTION\\EMOTION_DETECTION\\haarcascade_frontalface_default.xml')
model12 = load_model('C:\\Users\\cheth\\Desktop\\EMOTION_DETECTION\\EMOTION_DETECTION\\Emotion_little_vgg.h5')
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

classes12 = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

emotion_label = ['fear', 'joy', 'anger', 'guilt', 'disgust', 'shame', 'sadness']

c1 = ['neutral','calm','happy','surprised']

    
def suggest_remedy(mood):
    if mood in ('Happy','Neutral','Surprise'):
        return redirect_to_remedy(mood)
    else:
        return redirect_to_remedy(mood)

def redirect_to_remedy(mood):
    return render_template(mp.get(mood,'https://www.youtube.com/watch?v=F9wbogYwTVM'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict_mood():
    final_label = None
    cap = cv.VideoCapture(0)
    got = False
    while True:
        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)
            
            if(np.sum([roi_gray])!=0):
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
                preds = model12.predict(roi)[0]
                label = classes12[preds.argmax()]
                label_position = (x,y)
                final_label = label
                # got = True
                # break
                cv.putText(frame,label,label_position,cv.FONT_HERSHEY_COMPLEX,2,(0,255,0))
            else:
                cv.putText(frame,'No Face Found',(20,60),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255))
       
        cv.imshow('Emotion Detector',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    # print("Done")
    print(final_label)
    if final_label in ('Happy','Neutral','Surprise'):
        return render_template("happy.html")
    else:
        return render_template("sad.html")
    

def convert_class_to_emotion(pred):        
    label_conversion = {'0': 'neutral',
                        '1': 'calm',
                        '2': 'happy',
                        '3': 'sad',
                        '4': 'angry',
                        '5': 'fearful',
                        '6': 'disgust',
                        '7': 'surprised'}

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label
