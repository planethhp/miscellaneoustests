!if not exist "./files" mkdir files
# Download Face detection XML
!curl -L -o ./haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# Download emotion trained data
!curl -L -o ./emotion_model.hdf5 https://mechasolution.vn/source/blog/AI-tutorial/Emotion_Recognition/emotion_model.hdf5

import os
import shutil
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import csv


def detection(filename):
    img_source = "image/" + filename
    image = cv2.imread(img_source)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_defult.xml')
    emotion_classifier = load_model('emotion_model.hdf5', compile=False)
    EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30))

    if len(faces) > 0:
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        print(filename, label, emotion_personality, sep=',')


filenames = os.listdir('image')

for name in filenames:
    detection(name)