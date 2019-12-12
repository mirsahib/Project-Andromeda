# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:25:24 2019

@author: Mir Sahib
"""

import numpy as np
from numpy import linalg as LA
from scipy.stats import skew 
import pandas as pd
from os import listdir
from os.path import isfile, join 
import re
from scipy.spatial.distance import pdist
from numpy import linalg as LA

mypath = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\MSRAction3D_Skeleton_Joint_Extracted'
fileName = [f for f in listdir(mypath) if isfile(join(mypath, f))]
l = len(fileName)
for k in range(0,len(fileName)):
    label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file name
    content = np.loadtxt(open(mypath+"\\"+fileName[k], "rb"), delimiter=",", skiprows=1)
    matrix = np.zeros((20,content.shape[0],3))
    # reshape
    c = 0 #internal counter
    for q in range(0,20):
        matrix[q] = content[:,c:c+3]
        c=c+3
    # Both J1 and J2 are end site
    line_endSite = np.array([[19,9],[19,10],[19,15],[19,16],[9,10],[9,15],[9,16],[10,15],[10,16],[15,16]],np.int8)
    # distance between J1 and J2 are end site
    jj_distance_raw=np.array([])
    for i in range(10):
        for j in range(content.shape[0]):
            k = []
            k.append(matrix[line_endSite[i][0]][j])
            k.append(matrix[line_endSite[i][1]][j])
            jj_distance_raw=np.append(jj_distance_raw,pdist(k))
    jj_distance=np.reshape(jj_distance_raw,(content.shape[0],10))
    
    '''
    #2nd freature jj_orientation
    jj_orientation=np.array([])
    for i in range(frame_count-1):
        for j in range(10):
            jj_orientation=np.append(jj_orientation,(joint_vector[i][line_endSite[j][0]]-joint_vector[i][line_endSite[j][1]])/jj_distance[i][j])
                   
    jj_orientation=np.reshape(jj_orientation,(frame_count-1,10,3))
    '''
    #2nd freature jj_orientation
    jj_orientation=np.array([])
    for i in range(10):
        for j in range(content.shape[0]):
            jj_diff = matrix[line_endSite[i][0]][j]-matrix[line_endSite[i][1]][j]
            jj_norm = LA.norm(jj_diff)
            jj_orientation=np.append(jj_orientation,(jj_diff/jj_norm))
    
    print(jj_orientation)        
    
        
     