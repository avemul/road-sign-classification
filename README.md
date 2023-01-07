# Introduction
  An average of 2.5 million car crashes occur annually from which 1.3 million people die. Drunk and reckless drivers are the cause of around 10 thousand deaths each year in the United States alone.  With the implementation of autonomous cars, thousands of lives can be spared. Large corporations such as Tesla and Google, have worked around the idea of self driving cars  for quite some time with the use of machine learning concepts. Convolutional neural networks (CNN), can be used for the identification of various road signs, however the result must be extremely accurate as it is a matter of life and death. So, how effective are different CNN architectures for classifying a road sign as a traffic light, stop sign, speed limit or crosswalk sign?

# Background
A convolutional neural network (CNN), is a method of machine learning that is primarily used for image classification and visual tasks. CNNs are based on the visual cortex of the brain and several recent advancements have increased the use of CNNs in many upcoming projects. CNNs consist of several convolutional layers which an image is passed through to get an output. Convolution layers extract features from the image. Training data must be used for CNNs in order to adjust the weights and biases which contribute to the final accuracy. A convolutional neural network can be used for this project because the goal of the program is to classify road signs which is a visual task and can be achieved using a CNN. 
	A CNN can be implemented into a code using several methods however one of the most common is keras. Keras was developed by Google and is a deep learning API that can be used to facilitate the use of CNNs. Keras was written in python and can currently only be used in a python code. 

# Dataset and Technical tools
The dataset used for this code was taken from Kaggle. The name of the dataset is “Road Sign Detection” and was created by the user, LARXEL.

# Classification
### Neural architecture
### Walkthrough of code 
### Visual results
### Performance on the test dataset

# Analysis, Discussion
Overall, using a CNN model for this project produced good results with an accuracy of around 78%. The original code had an accuracy of 70% however adding an additional convolutional increased the accuracy of the code. CNN’s worked well in this context however the results were not as hoped. It is possible that the code can be improved by adding more layers however it is possible that a different method of machine learning would produce an increased accuracy score. All in all, the code can be improved using various methods however using a CNN is a good approach for decent results.


