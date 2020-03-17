#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:14:58 2020

@author: mirsahib
"""
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile,join
import re
import math
from histograms import *

def cart2sph(x, y, z,frame_rate):
    data = np.zeros([frame_rate,2])
    hxy = np.hypot(x, y)
    data[:,0] = np.arctan2(y, x) * (180/math.pi)
    data[:,1] = np.arctan2(z, hxy) * (180/math.pi)
    #data[:,2] = np.hypot(hxy, z)
    return data


def preData(jc,matrix):
    frame_rate = matrix.shape[1]
    numofjoints = len(jc) 
    data = np.zeros([numofjoints,frame_rate,2])
    for i in range(len(jc)):
        j = jc[i]
        data [i] = cart2sph(matrix[j][:,0],matrix[j][:,1],matrix[j][:,2],frame_rate)
    
    return data

def getStats(angles):
    alphaMean = np.zeros([angles.shape[0],angles.shape[1]])
    thetaMean = np.zeros([angles.shape[0],angles.shape[1]])
    for i in range(len(angles)):
        alphaMean[i]=angles[i][:,0]
        thetaMean[i]=angles[i][:,1]
    alphaDev = alphaMean
    thetaDev = thetaMean
    alphaMean = np.mean(alphaMean,axis=0)
    thetaMean = np.mean(thetaMean,axis=0)
    alphaDev = np.std(alphaDev,axis=0)
    thetaDev = np.std(thetaDev,axis=0)
    return [alphaMean,thetaMean,alphaDev,thetaDev]
    

def getJoints(jc,matrix):      
    joints = np.zeros([len(jc),matrix.shape[1],3])
    for i in range(len(jc)):
        joints[i] = matrix[jc[i]]
    return joints
        

def main():
    mypath = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/MSRAction3D_Skeleton_Joint_Extracted'
    fileName = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    fileName.sort()
    l = len(fileName)
    for k in range(0,l):
        label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file  name
        content = np.loadtxt(open(mypath+"/"+fileName[k], "rb"), delimiter=",", skiprows=1)
        matrix = np.zeros([20,content.shape[0],3])
        frame_rate = content.shape[0]
        # reshape
        c = 0 #internal counter
        for q in range(0,20):
            matrix[q] = content[:,c:c+3]
            c=c+3
        jc = np.array([8, 9, 12, 13, 14, 15, 18, 19, 20])# interested joint coordinate
        jc = jc-1
        sphericalData = preData(jc,matrix) 
        stats = getStats(sphericalData)
        joints = getJoints(jc,matrix)
        print('hello')
        
        
        
main()