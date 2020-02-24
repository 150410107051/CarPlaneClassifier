import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

DataDir ="/content/CarandPlane"
CateGories = ["car","plane"]

training_data = [ ]
IMG_SIZE =50
def craete_training_data():
  for category in CateGories:
    path = os.path.join(DataDir,category)
    class_num = CateGories.index(category)
    for img in os.listdir(path):
      try:
        img_array =cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
      except Exception as e:
        pass

craete_training_data()        

import random
random.shuffle(training_data)

for feature, label in training_data:
  X.append(feature)
  y.append(label) 
  
X= np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

//model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X =X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
