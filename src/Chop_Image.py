# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:35:25 2018

@author: Gustavo
"""
#%% Import libraries
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from IPython.display import display
from PIL import Image
import os
from scipy import ndimage, misc
import re

def Chop_Image(images_dir, wins_dir, prediction_image_name, win_length):

    #%% Load image    
    if re.search("\.(jpg|jpeg|png|bmp|tiff|tif)$", prediction_image_name):
        filepath = os.path.join(images_dir, prediction_image_name)
        I = misc.imread(filepath)
        if len(I.shape) > 2:            # If it is an RGB uint8 image
            I = I[:,:,0]                # Select first channel
            I = (I/255).astype(float)   # Convert uint8 to doube
    else:
        print('Image to chop was not found! Exiting')
        return
    
    #%% Show image
    plt.figure()
    plt.imshow(I, cmap='gray', interpolation='none')
    
    #%% Create Array of Windows
    pad_size = int( (win_length-1)/2 );
    I_padded = np.pad(I, pad_size, 'symmetric')
    win_array = np.zeros( (I_padded.size, win_length, win_length, 1) )
    
    counter = 0
    for (r,c), _ in np.ndenumerate(I):
        
        r_padded = r + pad_size
        c_padded = c + pad_size
        
        window = I_padded[r_padded-pad_size:r_padded+pad_size+1,c_padded-pad_size:c_padded+pad_size+1]
        
        win_array[counter,:,:,0] = window
        counter+=1

    np.save(wins_dir+'\\'+'prediction_array', win_array)
    
    return