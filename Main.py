# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:21:34 2018

@author: Gustavo
"""

#%% Clear Variables
from IPython import get_ipython 
get_ipython().magic('reset -sf')

#%% Import libraries
import numpy as np
import os
import sys

print(os.path.dirname(os.path.realpath(__file__)))
main_dir = os.getcwd() 
src_dir = main_dir +  "\src"
sys.path.insert(0, src_dir)

#%% Import functions
from Create_Dataset import Create_Dataset
from Synthesize_Dataset import Synthesize_Dataset
from Train import Train
from Test import Test
from Chop_Image import Chop_Image
from Predict import Predict

#%% Set general constants
data_dir = main_dir + '\data'
images_dir = data_dir + '\images'
wins_dir = data_dir + '\window arrays'
models_dir = data_dir + '\models'
res_dir = main_dir + '\\res'
#prediction_image_name = 'MAX_ca1_10 apical2_FIBER.tif'
prediction_image_name = 'predict_image1.png'
win_length = 25

#%% Create Dataset
Create_Dataset(images_dir, wins_dir, win_length)

#%% Synthesize Dataset
num_augmentations = 30
rotation_range = 360
Synthesize_Dataset(wins_dir, win_length, num_augmentations, rotation_range)

#%% Train
epochs = 20
batch_size = 64
k = 2
Train(wins_dir, models_dir, win_length, epochs, batch_size, k)

#%% Test
Test(wins_dir, models_dir, win_length, epochs, batch_size, k)

#%% Chop predicion image
Chop_Image(images_dir, wins_dir, prediction_image_name, win_length)

#%% Predict
Predict(images_dir, wins_dir, models_dir, res_dir, prediction_image_name)






