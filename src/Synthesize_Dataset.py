# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:40:41 2018

@author: Gustavo
"""
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')

def Synthesize_Dataset(wins_dir, win_length, num_augmentations, rotation_range):
    
    #%% Customize transformations
    datagen = ImageDataGenerator(
            rotation_range= rotation_range,
            horizontal_flip=True,
            fill_mode='nearest')

    #%% Load Data
    train_large_spine_win_array = np.load(wins_dir+'\\'+"train_large_spine_array.npy")
    train_large_nonspine_win_array = np.load(wins_dir+'\\'+"train_large_nonspine_array.npy")
    test_large_spine_win_array = np.load(wins_dir+'\\'+"test_large_spine_array.npy")
    test_large_nonspine_win_array = np.load(wins_dir+'\\'+"test_large_nonspine_array.npy")
    
    syn_train_length = (train_large_spine_win_array.shape[0])*num_augmentations
    syn_test_length = (test_large_spine_win_array.shape[0])*num_augmentations
    
    large_win_length = win_length + int(np.ceil( win_length/3 ))*2
    
    syn_train_large_spine_win_array = np.zeros((syn_train_length,1, 1, large_win_length, large_win_length ))
    syn_train_large_nonspine_win_array = np.zeros((syn_train_length,1, 1, large_win_length, large_win_length ))
    syn_test_large_spine_win_array = np.zeros((syn_test_length,1, 1, large_win_length, large_win_length ))
    syn_test_large_nonspine_win_array = np.zeros((syn_test_length,1, 1, large_win_length, large_win_length ))
    
    #%% Generate new Data
    # Train spine data
    
    for im in range(0,train_large_spine_win_array.shape[0]):
        image = train_large_spine_win_array[im,:,:].reshape(1, 1, large_win_length, large_win_length)
        i = 0
        for batch in datagen.flow(image,batch_size=1):
            syn_train_large_spine_win_array[im*num_augmentations+i, :,:,:] = batch
            i += 1
            if i >= num_augmentations:
                break  # otherwise the generator would loop indefinitely
                
    # Train nonspine data
    for im in range(0,train_large_nonspine_win_array.shape[0]):
        image = train_large_nonspine_win_array[im,:,:].reshape(1, 1, large_win_length, large_win_length)
        i = 0
        for batch in datagen.flow(image, batch_size=1):
            syn_train_large_nonspine_win_array[im*num_augmentations+i, :,:,:] = batch
            i += 1
            if i >= num_augmentations:
                break
     
    # Test spine data           
    for im in range(0,test_large_spine_win_array.shape[0]):
        image = test_large_spine_win_array[im,:,:].reshape(1, 1, large_win_length, large_win_length)
        i = 0
        for batch in datagen.flow(image, batch_size=1):
            syn_test_large_spine_win_array[im*num_augmentations+i, :,:,:] = batch
            i += 1
            if i >= num_augmentations:
                break
    
    # Test nonspine data  
    for im in range(0,test_large_nonspine_win_array.shape[0]):
        image = test_large_nonspine_win_array[im,:,:].reshape(1, 1, large_win_length, large_win_length)
        i = 0
        for batch in datagen.flow(image, batch_size=1):
            syn_test_large_nonspine_win_array[im*num_augmentations+i, :,:,:] = batch
            i += 1
            if i >= num_augmentations:
                break
    
    #%% Trim data to original size
    win_length_difference = large_win_length - win_length
        
    syn_train_spine_win_array = syn_train_large_spine_win_array[:,:,:,int(win_length_difference/2):int(large_win_length-win_length_difference/2),int(win_length_difference/2):int(large_win_length-win_length_difference/2)]
    syn_train_nonspine_win_array = syn_train_large_nonspine_win_array[:,:,:,int(win_length_difference/2):int(large_win_length-win_length_difference/2),int(win_length_difference/2):int(large_win_length-win_length_difference/2)]
    syn_test_spine_win_array = syn_test_large_spine_win_array[:,:,:,int(win_length_difference/2):int(large_win_length-win_length_difference/2),int(win_length_difference/2):int(large_win_length-win_length_difference/2)]
    syn_test_nonspine_win_array = syn_test_large_nonspine_win_array[:,:,:,int(win_length_difference/2):int(large_win_length-win_length_difference/2),int(win_length_difference/2):int(large_win_length-win_length_difference/2)]
    
    #%% Save synthetic data             
    np.save(wins_dir+'\\'+'syn_train_spine_array', syn_train_spine_win_array)
    np.save(wins_dir+'\\'+'syn_train_nonspine_array', syn_train_nonspine_win_array)
    np.save(wins_dir+'\\'+'syn_test_spine_array', syn_test_spine_win_array)
    np.save(wins_dir+'\\'+'syn_test_nonspine_array', syn_test_nonspine_win_array)
    
    return