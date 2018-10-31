# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:59:48 2018

@author: Gustavo
"""
import numpy as np
import matplotlib.pyplot as plt

def Plot_Predictions(predicted_classes, test_X, test_Y, win_length):

    # Show correct predictions
    correct = np.where(predicted_classes==test_Y)[0]
    print( "Found %d correct labels" % len(correct) )
    plt.figure()
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[correct].reshape(win_length,win_length), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
#        plt.tight_layout()

    # Show incorrect predictions
    incorrect = np.where(predicted_classes!=test_Y)[0]
    print( "Found %d incorrect labels" % len(incorrect) )
    plt.figure()
    for i, incorrect in enumerate(incorrect[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[incorrect].reshape(win_length,win_length), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
#        plt.tight_layout()
    return;