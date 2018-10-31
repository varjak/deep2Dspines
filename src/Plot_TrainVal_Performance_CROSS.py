# -*- coding: utf-8 -*-
"""
Created on Mon May 14 04:16:55 2018

@author: Gustavo
"""

import matplotlib.pyplot as plt

def Plot_TrainVal_Performance_CROSS(spine_train,k,counter):
    accuracy = spine_train.history['acc']
    val_accuracy = spine_train.history['val_acc']
    loss = spine_train.history['loss']
    val_loss = spine_train.history['val_loss']
    epochs = range(len(accuracy))
#    plt.figure()
    plt.subplot(k,2,(counter+1)*2-1)
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
#    plt.tight_layout()
#    plt.figure()
    plt.subplot(k,2,(counter+1)*2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
#    plt.tight_layout()
    return accuracy[-1], loss[-1], val_accuracy[-1], val_loss[-1]