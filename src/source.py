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
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd

"""
#functions
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
#model training lib
"""
import train
import read_mnist
import revimg

(x_train,y_train),(x_test,y_test) = read_mnist.read()

"""
testing and seeing images
"""

print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

"""
Normalizing dataset
"""
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"""
training model
"""
"""
model = train.train_model()
model.fit(x_train,y_train, epochs=10)

loss, acc = model.evaluate(x_test,y_test)
print ("loss is {} and accuracy is {}".format(loss,acc))

y_predict = model.predict (x_test)
y_pred = []

for val in range (0,10000):
    y_pred.append(np.argmax(y_predict[val]))
"""

"""
Inverting all images in the dataset
"""

x_test_opp = []
for img in range (len(x_test)):
    x_test_opp.append(revimg.reverse(x_test[img]))

x_test_opp = np.array(x_test_opp)


x_train_opp = []
for img in range (len(x_train)):
    x_train_opp.append(revimg.reverse(x_train[img]))

x_train_opp = np.array(x_train_opp)

model.fit (x_train_opp,y_train,epochs = 10)

loss_opp, acc_opp = model.evaluate (x_test_opp,y_test)


