## Introduction

An average of 2.5 million car crashes occur annually from which 1.3 million people die. Drunk and reckless drivers are the cause of around 10 thousand deaths each year in the United States alone.  With the implementation of autonomous cars, thousands of lives can be spared. Large corporations such as Tesla and Google, have worked around the idea of self-driving cars for a long time with the use of machine learning concepts. Convolutional neural networks (CNN) are often used for visual tasks and can be used for the identification of various road signs. So, how effective are different CNN architectures for classifying a road sign as a traffic light, stop sign, speed limit or crosswalk sign? 
## Background

Convolutional neural network (CNN) is a method of machine learning used for image classification and visual tasks. CNNs are based on the visual cortex of the brain and several recent advancements have increased the use of CNNs in many upcoming projects. CNNs consist of several convolutional layers which an image is passed through to get an output. Convolution layers extract features from the image. Training data must be used for CNNs to adjust the weights and biases which contribute to the final accuracy. A convolutional neural network can be used for this project because the goal of the program is to classify road signs which is a visual task and can be achieved using CNN.  

CNN can be implemented into a code using several methods, however one of the most common is keras. Keras was developed by Google and is a deep learning API that can be used to facilitate the use of CNNs. Keras was written in python and can currently only be used in python code.

## Dataset
The dataset used for this code was taken from Kaggle. The name of the dataset is “Road Sign Detection” and was created by the user, LARXEL.
## Classification
The following is the code we used for our final model, which had around 78-82% accuracy. 
```python
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
```
Import the necessary packages.

```python
DATADIR = "/Users/aadya/Downloads/archive"
CATEGORIES = ["images", "annotations"]
images = []
labels = []
imagedir = os.path.join(DATADIR, "images")
annotationdir = os.path.join(DATADIR, "annotations")
```
Provide the path to dataset and divide the data into two folders.
```python
width = 150
height = 200
```
Initialize the desired width and height for each image in the dataset
```python
for category in CATEGORIES:
   path = os.path.join(DATADIR, category)
   for img in os.listdir(path):
       img_arr = cv2.imread(os.path.join(path, img))
```
Join each image with its path.
```python
for img in os.listdir(imagedir):
   imagepath = os.path.join(imagedir, img)
   annotationpath = os.path.join(annotationdir, img.replace(".png", ".xml"))
   img_array = np.array(Image.open(imagepath).resize((width, height)))
   images.append(img_array)
# find the classification of each image by searching its corresponding annotation
   tree = ET.parse(annotationpath)
   root = tree.getroot()
   classification = root.find("./object/name").text
# set each classification as a value in an array and save the values in a list
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
```
Here, we match each image with its classification, and convert the classifications to their one-hot encoding. 
```python
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
```
Create the training, validation, and test data split. Also, normalize the image matrices.
```python
#create a stack of layers with the sequential model
n_classes = 4
model = Sequential()
#add input layer
model.add(InputLayer(input_shape=(height, width, 4)))
#add convolutional layer
model.add(Conv2D(15, kernel_size=(4, 4), strides=(3, 3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 1)))
#add convolutional layer
model.add(Conv2D(10, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 1)))
#flatten results from convolutional layer
model.add(Flatten())
#add hidden layer
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#add output layer
model.add(Dense(4, activation='softmax'))
#get results of model
model.summary()
#compile sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#train model in 20 epochs
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val))
```
Our model architecture starts with two convolutional layers, followed by two fully-connected hidden layers, and finally an output layer.
```python
#predict label of new set of data
y_test_prediction = model.predict(X_test)
correct_imgs = []
incorrect_imgs = []
incorrect_labels = []
#add images that were correctly labeled to a list
for y_pred, y_true, x in zip(y_test_prediction, y_test, X_test):
   if np.argmax(y_pred) == np.argmax(y_true):
       #correct prediction
       correct_imgs.append(x)
#add images that were incorrectly labeled to a list
   else:
       #incorrect prediction
       incorrect_imgs.append(x)
       incorrect_labels.append(y_pred)
#show three images that were correctly labeled
for i in range(3):
   Image.fromarray((correct_imgs[i] * 255).astype(np.uint8)).show()
#show three images that were incorrectly labeled
for i in range(3):
   Image.fromarray((incorrect_imgs[i] * 255).astype(np.uint8)).show()
   print(incorrect_labels[i])
```
Finally, we run the model on the test data and store some examples of images it labels correctly and incorrectly. Below, we display these examples.

Correctly Classified Images:

Incorrectly Classified Images:

Training and validation accuracy across 20 epochs:
```
Epoch 1/20
5/5 [==============================] - 1s 160ms/step - loss: 1.0893 - accuracy: 0.7064 - val_loss: 0.8169 - val_accuracy: 0.7614
Epoch 2/20
5/5 [==============================] - 0s 82ms/step - loss: 0.8192 - accuracy: 0.7471 - val_loss: 0.7882 - val_accuracy: 0.7614
Epoch 3/20
5/5 [==============================] - 0s 65ms/step - loss: 0.7979 - accuracy: 0.7455 - val_loss: 0.7993 - val_accuracy: 0.7614
Epoch 4/20
5/5 [==============================] - 0s 68ms/step - loss: 0.7749 - accuracy: 0.7504 - val_loss: 0.7773 - val_accuracy: 0.7500
Epoch 5/20
5/5 [==============================] - 0s 56ms/step - loss: 0.7271 - accuracy: 0.7537 - val_loss: 0.7499 - val_accuracy: 0.7500
Epoch 6/20
5/5 [==============================] - 0s 59ms/step - loss: 0.7005 - accuracy: 0.7651 - val_loss: 0.7302 - val_accuracy: 0.7500
Epoch 7/20
5/5 [==============================] - 0s 60ms/step - loss: 0.6713 - accuracy: 0.7798 - val_loss: 0.7073 - val_accuracy: 0.7386
Epoch 8/20
5/5 [==============================] - 0s 58ms/step - loss: 0.6449 - accuracy: 0.7847 - val_loss: 0.7069 - val_accuracy: 0.7386
Epoch 9/20
5/5 [==============================] - 0s 67ms/step - loss: 0.6118 - accuracy: 0.7977 - val_loss: 0.8284 - val_accuracy: 0.7273
Epoch 10/20
5/5 [==============================] - 0s 61ms/step - loss: 0.6605 - accuracy: 0.8042 - val_loss: 0.7231 - val_accuracy: 0.7500
Epoch 11/20
5/5 [==============================] - 0s 60ms/step - loss: 0.5858 - accuracy: 0.8271 - val_loss: 0.6947 - val_accuracy: 0.8068
Epoch 12/20
5/5 [==============================] - 0s 60ms/step - loss: 0.5301 - accuracy: 0.8303 - val_loss: 0.6853 - val_accuracy: 0.7955
Epoch 13/20
5/5 [==============================] - 0s 62ms/step - loss: 0.4889 - accuracy: 0.8369 - val_loss: 0.6999 - val_accuracy: 0.8068
Epoch 14/20
5/5 [==============================] - 0s 60ms/step - loss: 0.4638 - accuracy: 0.8548 - val_loss: 0.6556 - val_accuracy: 0.8068
Epoch 15/20
5/5 [==============================] - 0s 54ms/step - loss: 0.4377 - accuracy: 0.8777 - val_loss: 0.7396 - val_accuracy: 0.8068
Epoch 16/20
5/5 [==============================] - 0s 55ms/step - loss: 0.3965 - accuracy: 0.8842 - val_loss: 0.6655 - val_accuracy: 0.8182
Epoch 17/20
5/5 [==============================] - 0s 54ms/step - loss: 0.3697 - accuracy: 0.8825 - val_loss: 0.7157 - val_accuracy: 0.8068
Epoch 18/20
5/5 [==============================] - 0s 57ms/step - loss: 0.3290 - accuracy: 0.8940 - val_loss: 0.6621 - val_accuracy: 0.8068
Epoch 19/20
5/5 [==============================] - 0s 56ms/step - loss: 0.2924 - accuracy: 0.9119 - val_loss: 0.7267 - val_accuracy: 0.8182
Epoch 20/20
5/5 [==============================] - 0s 57ms/step - loss: 0.2583 - accuracy: 0.9217 - val_loss: 0.7977 - val_accuracy: 0.7955
```

## Analysis
Overall, using a CNN model for this project produced satisfactory results with an accuracy of 78%-82%. The original code had an accuracy of 72%, however adding an additional convolutional layer increased the accuracy of the code. CNNs worked well in this context and the results were as imagined. The code can be improved by adding more layers; however, a different method of machine learning could produce an increased accuracy score. Another way the accuracy of the code can be improved is by using a larger dataset. A larger dataset would increase the amount of training data which can lead to improved results. In the end, the code can be improved using various methods, however using CNN is one good approach for adequate results. 

