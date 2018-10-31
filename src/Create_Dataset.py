# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:21:45 2018

@author: Gustavo
"""

import numpy as np
import os
from scipy import ndimage, misc
import re
import matplotlib.pyplot as plt
import random
from PIL import Image


def Create_Dataset(images_dir, wins_dir, win_length):
# The interactive quiz in this code was tested by showing every image in the Spyder's console,
# inline with the text. This option is recommended to force images to show during execution time.
    
    #%% Load images
    I_list = []
    valid_extensions = [".jpg",".gif",".png",".tif", ".tiff"]
    for f in os.listdir(images_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_extensions:
            continue
        I = np.array(Image.open(os.path.join(images_dir,f)))
        I_list.append(I)
        
    Nimages = len(I_list)
    
    #%% Show images
    for i in range(0, Nimages):
        plt.figure()
        plt.imshow(I_list[i], cmap='gray')
    
    #%% Trim images (so windows can be later placed without restriction in the untrimmed image)
    win_semi_length = int( np.floor(win_length/2) )
    extra_length = int(np.ceil( win_length/3 ))
    bias_intensity = 0.1
    num_high = 0
    
    for i in range(0, Nimages):
        I_shape = I_list[i].shape
        I_trimmed = I_list[i][win_semi_length+extra_length:I_shape[0]- (win_semi_length+extra_length), win_semi_length+extra_length:I_shape[1]-(win_semi_length+extra_length)]
        num_high = num_high + np.sum(I_trimmed > bias_intensity) 
    
    rois_subs = np.empty((len(I_list),4))
    
    for i in range(0, len(I_list)):
        rois_subs[i,:] = np.array([[(win_semi_length+extra_length),np.shape(I_list[i])[1]-(win_semi_length+extra_length), (win_semi_length+extra_length),np.shape(I_list[i])[1]-(win_semi_length+extra_length)]], dtype=int)
    
    rois_subs = rois_subs.astype(int)
    
    #%% Divide subs by wether pixel is above or below threshold (to make number of dark pixel picks reasonable)
    total_low_subs = np.empty((0,3), dtype=int)
    total_high_subs = np.empty((0,3), dtype=int)

    for i in range(0, len(I_list)):
        low_subs = np.empty((0,2), dtype=int)
        high_subs = np.empty((0,2), dtype=int)
        I = I_list[i]
        
        for j in range(0, rois_subs[i].shape[0]):
            roi =I[rois_subs[i,2]:rois_subs[i,3], rois_subs[i,0]:rois_subs[i,1]]
            roi_low_subs = np.argwhere(roi<bias_intensity) + np.array([rois_subs[i,2], rois_subs[i,0]])
            roi_high_subs = np.argwhere(roi>bias_intensity) + np.array([rois_subs[i,2], rois_subs[i,0]])
            
            low_subs = np.concatenate((low_subs, roi_low_subs), axis=0)
            high_subs = np.concatenate((high_subs, roi_high_subs), axis=0)
            
        # For image i   
        low_subs = np.concatenate((low_subs, np.ones((low_subs.shape[0],1))*i ), axis=1)
        high_subs = np.concatenate((high_subs, np.ones((high_subs.shape[0],1))*i ), axis=1)
        
        # For all images
        total_low_subs = np.concatenate((total_low_subs, low_subs), axis=0)
        total_high_subs = np.concatenate((total_high_subs, high_subs), axis=0)
     
    total_low_subs = total_low_subs.astype(int)   
    total_high_subs = total_high_subs.astype(int)
    
    chosen_subs = np.array([0], dtype=int)     # so it doesnt have size zero
    choice = 0  # initialize to prevent errors
    
    #%% Quiz
    
    # Initialize window and classification arrays
    ground_truth_onehot = np.ones((num_high,), dtype=int)*-1
    large_win_array = np.zeros((num_high, win_length + 2*extra_length, win_length + 2*extra_length ))
    
    # Start interactive quiz
    for i in range(0,int( total_low_subs.shape[0] + total_high_subs.shape[0]) ):
        
        # Chose the high or low intensity group of pixels, based on a probability
        if chosen_subs.size == 0:
            choice = np.logical_not(choice.astype(bool))
        else:
            choice = np.random.choice(2, 1, p=[0.1,0.9])
            
        if (choice):
            chosen_subs = np.copy(total_high_subs)
        else:
            chosen_subs = np.copy(total_low_subs)
            
        # Chose random pixel from the group
        random_pool_idx = random.randint(0,chosen_subs.shape[0])
        
        row = chosen_subs[random_pool_idx,][0]
        col = chosen_subs[random_pool_idx,][1]
        im = chosen_subs[random_pool_idx,][2]
        
        # Delete it from the group
        chosen_subs = np.delete(chosen_subs, (random_pool_idx), axis=0)
        
        if (choice):
            total_high_subs = np.copy(chosen_subs)
        else:
            total_low_subs = np.copy(chosen_subs)
        
        # Get image of chosen pixel
        I = I_list[im]
        
        # Extract windows centered in chosen pixel:
        # - one with the predefined size (for visualization)
        # - one with a larger size (for data augmentation)
        win = I[row-win_semi_length:row+win_semi_length+1, col-win_semi_length:col+win_semi_length+1]
        large_win = I[row-(win_semi_length+extra_length) : row+(win_semi_length+extra_length)+1, col-(win_semi_length+extra_length) : col+(win_semi_length+extra_length)+1]
        
        # Show window with predefined size
        print('Spines: '+str( np.count_nonzero(ground_truth_onehot>0) ) )
        plt.figure()
        plt.imshow(win, cmap='gray')
        plt.plot(win_semi_length, win_semi_length, 'r+')
        plt.show()
        
        # Ask user to classify it
        spine_or_not_str = input("Spine (1)? Not (0)? Skip (2)? End (3)? ")
        
        if not spine_or_not_str:
            spine_or_not = 2
        else:                    
            spine_or_not = int(spine_or_not_str, 10)
        
        if (spine_or_not == 0) or (spine_or_not == 1):
            # Store large window
            ground_truth_onehot[i] = spine_or_not
            large_win_array[i] = large_win
        elif (spine_or_not == 2):
            continue
        else:
            break
        
    #%% Remove array's unused positions
    onehot_negative_idxs = np.where(ground_truth_onehot < 0)[0]
    large_win_array_elite = np.delete(large_win_array, onehot_negative_idxs, 0)
    ground_truth_onehot_elite = np.delete(ground_truth_onehot, onehot_negative_idxs, 0)
    
    #%% Randomly delete non-spine windows so to equal spine windows
    onehot_positive_idxs = np.where(ground_truth_onehot_elite > 0)[0]
    onehot_zero_idxs = np.where(ground_truth_onehot_elite == 0)[0]
    
    random_onehot_zero_idxs_positions = random.sample(range(0, onehot_zero_idxs.shape[0]), onehot_positive_idxs.shape[0])
    onehot_zero_idxs_just = onehot_zero_idxs[random_onehot_zero_idxs_positions]
    
    large_spine_win_array = large_win_array_elite[onehot_positive_idxs]
    large_nonspine_win_array = large_win_array_elite[onehot_zero_idxs_just]
    
    #%% Split data into train and test
    num_spines_samples = onehot_positive_idxs.shape[0]
    train_test_fraction = 0.75
    num_spines_samples_train = int( np.round(train_test_fraction*num_spines_samples) )
    #
    train_large_spine_win_array = large_spine_win_array[0:num_spines_samples_train]
    train_large_nonspine_win_array = large_nonspine_win_array[0:num_spines_samples_train]
    
    test_large_spine_win_array = large_spine_win_array[num_spines_samples_train:]
    test_large_nonspine_win_array = large_nonspine_win_array[num_spines_samples_train:]
    
    #%% Save datasets
    np.save(wins_dir+'\\'+'train_large_spine_array', train_large_spine_win_array)
    np.save(wins_dir+'\\'+'train_large_nonspine_array', train_large_nonspine_win_array)
    np.save(wins_dir+'\\'+'test_large_spine_array', test_large_spine_win_array)
    np.save(wins_dir+'\\'+'test_large_nonspine_array', test_large_nonspine_win_array)
    
    return
    