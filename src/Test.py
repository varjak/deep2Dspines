# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:51:23 2018

@author: Gustavo
"""

#%% Import libraries
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from IPython.display import display
from PIL import Image
from scipy import ndimage, misc
from sklearn.model_selection import StratifiedKFold
import os

#%% Import functions
from Plot_Predictions import Plot_Predictions
from Plot_TrainVal_Performance_CROSS import Plot_TrainVal_Performance_CROSS


def Test(wins_dir, models_dir, win_length, epochs, batch_size, k):
    
    #%% Load Data
    test_spine_array = np.load(wins_dir+'\\'+'syn_test_spine_array.npy')
    test_nonspine_array = np.load(wins_dir+'\\'+'syn_test_nonspine_array.npy')  
        
    test_spine_array=test_spine_array[:,0,0,:,:].reshape(test_spine_array.shape[0], win_length, win_length, 1)
    test_nonspine_array=test_nonspine_array[:,0,0,:,:].reshape(test_nonspine_array.shape[0], win_length, win_length, 1)
    
    test_length = test_spine_array.shape[0]
    
    test_X = np.concatenate( (test_spine_array,test_nonspine_array), axis=0)
    test_Y = np.concatenate( (np.ones( (test_length,) , dtype=int),  np.zeros( (test_length,) , dtype=int) ), axis=0)
    test_Y_one_hot = to_categorical(test_Y)
    
    #%% Load Model
    model = load_model(models_dir+'\\'+'model1.h5py')
    
    #%% Evaluate Testing
    test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])    
    
    #%% Test
    predicted_classes = model.predict(test_X)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    
    #%% Show Predictions
    Plot_Predictions(predicted_classes, test_X, test_Y, win_length)
    
    #%% Report Classification
    num_classes = 2
    from sklearn.metrics import classification_report
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(test_Y, predicted_classes, target_names=target_names))

    return