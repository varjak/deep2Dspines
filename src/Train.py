# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:29:39 2018

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
from Plot_TrainVal_Performance_CROSS import Plot_TrainVal_Performance_CROSS

#%% Network
def Create_NN(win_length):
# Architecture:
# Conv 32 3x3 relu -> Max 2x2 -> 
# Conv 32 3x3 relu -> Max 2x2 -> 
# Conv 128 3x3 relu -> Max 2x2 -> 
# Flatten -> Dense 128 relu -> Dense 1 sigmoid
    
    spine_model = None
    num_classes = 2
    
    spine_model = Sequential()
    spine_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(win_length,win_length,1),padding='same'))
    spine_model.add(MaxPooling2D((2, 2),padding='same'))

    spine_model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    spine_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    spine_model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))                
    spine_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    
    spine_model.add(Flatten())
    spine_model.add(Dense(128, activation='relu'))                
    spine_model.add(Dense(num_classes, activation='sigmoid'))
    
    spine_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    return spine_model

#%% Train Network
def Train(wins_dir, models_dir, win_length, epochs, batch_size, k):
    
    #%% Set constants
    num_classes = 2     
    
    #%% Load Data
    train_spine_array = np.load(wins_dir+'\\'+'syn_train_spine_array.npy')
    train_nonspine_array = np.load(wins_dir+'\\'+'syn_train_nonspine_array.npy')  
        
    train_spine_array=train_spine_array[:,0,0,:,:].reshape(train_spine_array.shape[0], win_length, win_length, 1)
    train_nonspine_array=train_nonspine_array[:,0,0,:,:].reshape(train_nonspine_array.shape[0], win_length, win_length, 1)
    
    train_length = train_spine_array.shape[0]

    train_X = np.concatenate( (train_spine_array,train_nonspine_array), axis=0)
    train_Y = np.concatenate( (np.ones( (train_length,) , dtype=int),  np.zeros( (train_length,) , dtype=int) ), axis=0)
    train_Y_one_hot = to_categorical(train_Y)

    #%% Train and Cross-Validate
    batch_size = 64
    epochs = 20
    seed=13
    k = 2
    # Define k-fold cross-validation
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    plt.figure()
    counter = 0
    val_acc_vec = np.zeros(k)
    
    for train_idxs, val_idxs in kfold.split(train_X, train_Y):
        
        print("Cross-Validation {}/{}".format(counter+1, k))
        
        spine_model = Create_NN(win_length)

        # Train
        spine_train = spine_model.fit(train_X[train_idxs], train_Y_one_hot[train_idxs], batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(train_X[val_idxs], train_Y_one_hot[val_idxs]))
        
        # Plot performance
        accuracy, loss, val_accuracy, val_loss = Plot_TrainVal_Performance_CROSS(spine_train,k,counter)
        
        # Evaluate Training
        val_acc_vec[counter] = val_accuracy * 100
        
        counter+=1
      
    print(val_acc_vec)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(val_acc_vec), np.std(val_acc_vec)))
        
    #%% Train with whole training data
    spine_model.fit(train_X, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1)
    spine_model.summary()
        
    spine_model.save(models_dir+'\\'+'model1.h5py')        
    
    return
    
    
    