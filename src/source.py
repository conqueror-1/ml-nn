#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sankalp (IDRP19CG201)
"""

"""
#libs
"""

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd

"""
#functions
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

"""
#model training lib
"""
import train
import read_mnist

(x_train,y_train),(x_test,y_test) = read_mnist.read()

"""
testing and seeing images
"""
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

model = train.train_model() 
