import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
DATADIR = "/Users/Aditya Saraf/Downloads/archive"
CATEGORIES = ["images", "annotations"]

images = []
labels = []
# loop over all images
imagedir = os.path.join(DATADIR, "images")
annotationdir = os.path.join(DATADIR, "annotations")
for img in os.listdir(imagedir):
    imagepath = os.path.join(imagedir, img)
    annotationpath = os.path.join(annotationdir, img.replace(".png", ".xml"))
    img_array = np.array(Image.open(imagepath).resize((300, 400)))
    images.append(img_array)

    tree = ET.parse(annotationpath)
    root = tree.getroot()
    classification = root.find("./object/name").text
    # trafficlight = [1, 0, 0, 0], speedlimit = [0, 1, 0, 0], crosswalk = [0, 0, 1, 0], stop = [0, 0, 0, 1]
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
# figure out what the rest does in specific
X = np.asarray(images)
y = np.asarray(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33)
# final split: 70% train, ~20% val, ~10% test

X_train = X_train.reshape(X_train.shape[0], 400, 300, 4)
X_val = X_val.reshape(X_val.shape[0], 400, 300, 4)
X_test = X_test.reshape(X_test.shape[0], 400, 300, 4)
X_train = X_train.astype('float16')
X_val = X_val.astype('float16')
X_test = X_test.astype('float16')
X_train /= 255
X_val /= 255
X_test /= 255
n_classes = 4
# print("Shape before one-hot encoding: ", y_train.shape)
# y_train = np_utils.to_categorical(y_train, n_classes)
# y_test = np_utils.to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", y_train.shape)
model = Sequential()
model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(400, 300, 4)))
model.add(MaxPool2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_val, y_val))
# print(X.shape)
# # X's shape: (877, 400, 300, 4). This means 877 images, with each image being represented as a 400x300 array with 4
# # channels (CMYK)
# Y = np.asarray(labels)
# print(Y.shape)
# # Y's shape: (877, 4). This means 877 labels, where each label is a vector of 4 numbers (you can
# # think of each number as the probabilities of the image being each of the 4 categories).
# print(labels)

# Use the trained model to label the test set.
# predictions = model.predict(X_test)
# TODO: for the writeup, collect 5 labels (a mix of correct and incorrect examples) and their corresponding images