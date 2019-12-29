#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process some recordings of key presses to recover characters.
Created on Fri Dec 13 22:44:38 2019

@author: charles
"""

import argparse
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import fftconvolve
import os
from keras.models import load_model,Sequential
from keras.layers import Dense,Conv1D,Flatten,Reshape
import matplotlib.pyplot as plt

import split_audio

def get_output(predictions,class_names):
    for i in range(len(class_names)):
        print(class_names[i],end="\t")
    print("")
    
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            print("%02.0f"%(predictions[i,j]*100)+"%",end="\t")
        print("-> " + class_names[np.argmax(predictions[i])],end="\n")

def get_output_plt(predictions,class_names):
    for i in range(len(predictions)):
        plt.subplot(1,len(predictions),i+1)
        plt.yticks([])
        #plt.xticks(rotation=90)
        plt.grid(False)
        plt.bar(class_names, predictions[i])

def main():
    parser = argparse.ArgumentParser(description='Process some recordings of key presses to recover characters.')
    parser.add_argument('--train-path', type=str, help='Training directory containing files KEY_number.waw')
    parser.add_argument('--test-path', type=str, help='File to predict.')
    parser.add_argument('--method', type=str, default='ml_mlp', help='Method to use: ml_mlp/ml_cnn/cross_correlation/fft')
    parser.add_argument('--model', type=str, default='', help='Path to save/load H5 model')
    parser.add_argument('--trigger', type=float, default=1000, help='Trigger threshold')
    parser.add_argument('--lbound-samples', type=float, default=1050, help='Lower sample bound')
    parser.add_argument('--ubound-samples', type=float, default=2000, help='Upper sample bound')
    args = parser.parse_args()

    # Args
    train_path = args.train_path
    test_path = args.test_path
    method = args.method
    model_path = args.model

    # Open WAV train set
    train_inputs = []
    train_labels = []
    class_names = []

    for filename in os.listdir(train_path):
        if filename.endswith(".wav"):
            train_inputs.append(split_audio.normalize(read(train_path + "/" + filename)[1]))
            label = str.split(filename,"_")[0]
            if (label not in class_names):
                class_names.append(label)
            train_labels.append(class_names.index(label))

    test_inputs = split_audio.split_file(file=test_path,normalize_result=True, trigger=args.trigger)[0]

    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    test_inputs = np.array(test_inputs)

    train_inputs = train_inputs[:,args.lbound_samples:args.ubound_samples]
    test_inputs = test_inputs[:,args.lbound_samples:args.ubound_samples]
        
    # DEBUG : show sample of input
    #for i in range(25):
    #    plt.subplot(5,5,i+1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.grid(False)
    #    plt.xlabel(class_names[train_labels[i]])
    #    plt.plot(train_inputs[i])
    #plt.show()
    #
    #for i in range(3):
    #    plt.subplot(5,5,i+1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.grid(False)
    #    plt.plot(test_inputs[i])
    #plt.show()

    if(method == 'ml_mlp' or method == 'ml_cnn'):
        if(not os.path.isfile(model_path)):
            model = Sequential()
            
            if(method == 'ml_mlp'):
                model.add(Dense(len(class_names)*500, input_dim=train_inputs[0].size, activation='relu'))
                model.add(Dense(len(class_names)*500, activation='relu'))
                model.add(Dense(len(class_names)*10, activation='relu'))
                model.add(Dense(len(class_names)*5, activation='relu'))
                model.add(Dense(len(class_names)*5, activation='relu'))
                model.add(Dense(len(class_names)*5, activation='relu'))
                model.add(Dense(len(class_names), activation='softmax'))

                # compile and train the keras model
                model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
                model.fit(train_inputs, train_labels, epochs=25, validation_split=0.1)
                
            if(method == 'ml_cnn'):
                # FIXME : network not properly tuned
                model.add(Reshape((train_inputs.shape[1], 1), input_shape=(train_inputs.shape[1], )))
                model.add(Conv1D(10,10,activation='relu'))
                model.add(Flatten())
                model.add(Dense(len(class_names)*16, activation='relu'))
                model.add(Dense(len(class_names)*8, activation='relu'))
                model.add(Dense(len(class_names)*4, activation='relu'))
                model.add(Dense(len(class_names), activation='relu'))
                model.add(Dense(len(class_names), activation='softmax'))

                # compile and train the keras model
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
                model.fit(train_inputs, train_labels, epochs=100, validation_split=0.1)
                
            if(model_path != ""): model.save(model_path)
            
        else:
            model = load_model(model_path)

        predictions = model.predict(test_inputs)
        get_output(predictions,class_names)
        
    elif(method == 'cross_correlation'):
        trains = []
        for i in range(len(train_inputs)):
            trains.append(np.fft.rfft(train_inputs[i]))
        tests = []
        for j in range(len(test_inputs)):
            tests.append(np.fft.rfft(test_inputs[j]))
            
        for i in range(len(train_inputs)):
            print (class_names[train_labels[i]], end="\t")
            for j in range(len(test_inputs)):
                dist = np.average(np.abs(np.multiply(trains[i],tests[j])))
                print(np.average(np.abs(dist)),end="\t")
            print("")
            
    elif(method == 'fft'):
        trains = []
        for i in range(len(train_inputs)):
            trains.append(np.log10(np.abs(np.fft.rfft(train_inputs[i]))))
        tests = []
        for j in range(len(test_inputs)):
            tests.append(np.log10(np.abs(np.fft.rfft(test_inputs[j]))))
            
        for i in range(len(train_inputs)):
            print (class_names[train_labels[i]], end="\t")
            for j in range(len(test_inputs)):
                print(np.average(np.abs(tests[j]-trains[i])),end="\t")
            print("")          
            
    else:
        print ("Invalid method")
    
if __name__== "__main__":
  main()