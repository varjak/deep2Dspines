# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:59:58 2018

@author: Gustavo
"""

#%% Import libraries
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage, misc
import re
import scipy.misc

def Predict(images_dir, wins_dir, models_dir, res_dir, prediction_image_name):
    
    #%% Load image    
    if re.search("\.(jpg|jpeg|png|bmp|tiff|tif)$", prediction_image_name):
        filepath = os.path.join(images_dir, prediction_image_name)
        I_shape = misc.imread(filepath).shape[0:2]
     
    #%% Load model
    model = load_model(models_dir + '\\' + 'model1.h5py')
    
    #%% Load prediction array
    window_array = np.load(wins_dir+'\\'+'prediction_array.npy')

    predicted_classes = model.predict(window_array)
    spine_prob_vec = predicted_classes[:,1]
    prob_image = np.resize(spine_prob_vec, I_shape)
    
    #%% Show figure
    plt.figure()
    plt.imshow(prob_image, cmap='gray')
    
    #%% Save image
    scipy.misc.imsave(res_dir + '\\' + 'res1.png', prob_image)
    
    return
        