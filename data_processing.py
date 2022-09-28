import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
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

X = np.asarray(images)
print(X.shape)
# X's shape: (877, 400, 300, 4). This means 877 images, with each image being represented as a 400x300 array with 4
# channels (CMYK)
Y = np.asarray(labels)
print(Y.shape)
# Y's shape: (877, 4). This means 877 labels, where each label is a vector of 4 numbers (you can
# think of each number as the probabilities of the image being each of the 4 categories).
print(labels)