#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:30:59 2019

@author: charles
"""

import argparse
import numpy as np
from scipy.io.wavfile import read, write
import os

def normalize(x: np.ndarray):
    return x/np.max(x);

def split_file(file, time_before=0.025, time_after=0.2, trigger=1000, normalize_result=False):
    outputs = []

    # Open file
    sample_rate, a = read(file)
    a = np.array(a,dtype=float)
    
    # Compute params
    length_after = (int)(time_after*sample_rate)
    length_before = (int)(time_before*sample_rate)
    
    # Display sound (debug)
    #plt.plot(a)
    #plt.show()
    
    i = 0
    while i < a.size :
        # End of usable recording
        if(i+length_after > a.size):
            break;
        if (a[i] > trigger and i >= length_before):
            sub = a[i-length_before:i+length_after]
            if(normalize_result): sub = normalize(sub)
            outputs.append(sub)
            i += length_after
        i += 1
    
    return outputs, sample_rate;

def main():
    parser = argparse.ArgumentParser(description='Split key presses recording.')
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--out-dir', type=str, help='Output directory')
    parser.add_argument('--label', type=str, help='Output files prefix')
    parser.add_argument('--split-label-char', type=str, default='', help='Char to split the label string')
    parser.add_argument('--trigger', type=float, default=1000, help='Trigger threshold')
    parser.add_argument('--time_before', type=float, default=0.025, help='Samples to keep before triggers (s)')
    parser.add_argument('--time_after', type=float, default=0.2, help='Samples to keep after triggers (s)')
    args = parser.parse_args()
    
    outputs,sample_rate = split_file(args.input,args.time_before, args.time_after, args.trigger)
    
    if(args.split_label_char == ''):
        labels = [args.label]
    else:
        labels = str.split(args.label, args.split_label_char) 
    
    if(len(outputs)%len(labels)):
        print("ERROR!")
        return
        
    n = 0; i = 0
    for output in outputs:
        while os.path.isfile(args.out_dir + "/" + labels[i%len(labels)] + "_" + str(n) + ".wav"):
            n+=1
        write(args.out_dir + "/" + labels[i%len(labels)] + "_" + str(n) + ".wav", sample_rate, np.asarray(output, dtype=np.int16))
        print('Created ' + args.out_dir + "/" + labels[i%len(labels)] + "_" + str(n) + ".wav!")
        i += 1
        
if __name__== "__main__":
  main()
