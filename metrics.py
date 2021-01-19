# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OPb6BqHzVZJQ6xoJr_qkr5-u7MoALlXy
"""

def mse(orig, pred):
    sum = 0
    for i in range(len(orig)):
        sum += (pred[i] - orig[i]) ** 2
    mse = sum/len(orig)
    return mse

def r2(orig, pred):
    SSres = 0
    SStot = 0
    yt = 0
    for i in range(len(orig)):
        yt += orig[i]
    yt = yt/len(orig)
    for i in range(len(orig)):
        SSres += (pred[i] - orig[i]) ** 2
    for i in range(len(orig)):
        SStot += (orig[i] - yt) ** 2
    
    return SSres/SStot