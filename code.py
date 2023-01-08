import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
DATADIR = "/Users/aadya/Downloads/archive"
CATEGORIES = ["images", "annotations"]
images = []
labels = []
imagedir = os.path.join(DATADIR, "images")
annotationdir = os.path.join(DATADIR, "annotations")
width = 150
height = 200
for category in CATEGORIES:
   path = os.path.join(DATADIR, category)
   for img in os.listdir(path):
       img_arr = cv2.imread(os.path.join(path, img))
for img in os.listdir(imagedir):
   imagepath = os.path.join(imagedir, img)
   annotationpath = os.path.join(annotationdir, img.replace(".png", ".xml"))
   img_array = np.array(Image.open(imagepath).resize((width, height)))
   images.append(img_array)
   tree = ET.parse(annotationpath)
   root = tree.getroot()
   classification = root.find("./object/name").text
classificationarr = []
   if classification == 'trafficlight':
       classificationarr = [1, 0, 0, 0]
   elif classification == 'speedlimit':
       classificationarr = [0, 1, 0, 0]
   elif classification == 'crosswalk':
       classificationarr = [0, 0, 1, 0]
   elif classification == 'stop':
       classificationarr = [0, 0, 0, 1]
   labels.append(classificationarr)
X = np.asarray(images)
y = np.asarray(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
X_train = X_train.astype('float16')
X_val = X_val.astype('float16')
X_test = X_test.astype('float16')
X_train /= 255
X_val /= 255
X_test /= 255
n_classes = 4
model = Sequential()
model.add(InputLayer(input_shape=(height, width, 4)))
model.add(Conv2D(15, kernel_size=(4, 4), strides=(3, 3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 1)))
model.add(Conv2D(10, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val))
y_test_prediction = model.predict(X_test)
correct_imgs = []
incorrect_imgs = []
incorrect_labels = []
for y_pred, y_true, x in zip(y_test_prediction, y_test, X_test):
   if np.argmax(y_pred) == np.argmax(y_true):
       correct_imgs.append(x)
   else:
       incorrect_imgs.append(x)
       incorrect_labels.append(y_pred)
for i in range(3):
   Image.fromarray((correct_imgs[i] * 255).astype(np.uint8)).show()
for i in range(3):
   Image.fromarray((incorrect_imgs[i] * 255).astype(np.uint8)).show()
   print(incorrect_labels[i])
