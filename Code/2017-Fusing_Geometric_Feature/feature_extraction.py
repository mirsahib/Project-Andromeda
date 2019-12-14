# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:25:24 2019

@author: Mir Sahib
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd
from os import listdir
from os.path import isfile, join 
import re
from scipy.spatial.distance import pdist
import warnings


mypath = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\MSRAction3D_Skeleton_Joint_Extracted'
fileName = [f for f in listdir(mypath) if isfile(join(mypath, f))]
l = len(fileName)
for k in range(0,len(fileName)):
    label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file name
    content = np.loadtxt(open(mypath+"\\"+fileName[k], "rb"), delimiter=",", skiprows=1)
    matrix = np.zeros((20,content.shape[0],3))
    frame_rate = content.shape[0]
    # reshape
    c = 0 #internal counter
    for q in range(0,20):
        matrix[q] = content[:,c:c+3]
        c=c+3
        
        # J1 and J2 directly adjacent in the kinectic chain
    lines = np.array([[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],
        [10,11],[11,12],[1,13],[13,14],[14,15],[15,16],[1,17],[17,18],[18,19],[19,20]])

    # J1 is end site, J2 is two steps away
    lines = np.append(lines,[[4,2],[5,7],[9,11],[13,15],[17,19]],axis=0)

    # Both J1 and J2 are end site
    lines = np.append(lines, [[4,7],[4,11],[4,15],[4,19],[7,11],
        [7,15],[7,19],[12,15],[12,19],[15,19]], axis=0)
    lines = lines - 1
    numOfLine = lines.shape[0]
    #1st feature distance between J1 and J2 are end site
    jj_distance_raw=np.array([])
    for i in range(numOfLine):
        for j in range(frame_rate):
            s = []
            s.append(matrix[lines[i][0]][j])
            s.append(matrix[lines[i][1]][j])
            jj_distance_raw=np.append(jj_distance_raw,pdist(s))
    jj_distance=np.reshape(jj_distance_raw,(frame_rate,numOfLine))
    
    #2nd freature jj_orientation
    jj_orientation=np.zeros((frame_rate,3*numOfLine))
    for i in range(lines.shape[0]):
        jj_orientation_raw = np.array([])
        for j in range(frame_rate):
            jj_diff = matrix[lines[i][0]][j]-matrix[lines[i][1]][j]
            jj_norm = LA.norm(jj_diff)
            #jj_orientation_raw=np.append(jj_orientation_raw,(jj_diff/jj_norm))
            try:
                jj_orientation_raw=np.append(jj_orientation_raw,(jj_diff/jj_norm))
            except warnings:
                jj_orientation_raw=np.append(jj_orientation_raw,(0,0,0))

        jj_orientation[:,i*3:(i*3)+3] = np.reshape(jj_orientation_raw,(frame_rate,3))
  
    #3rd feature joint line distance
    plane_joint=np.array([[0,7,9],[1,8,10],[19,2,3],[4,13,15],[5,14,16]])
    jl_distance=np.zeros((frame_rate,5))
    ll_angle=np.array([])
    for i in range(5):
        for j in range(frame_rate):
            a=np.array(matrix[plane_joint[i][0]][j])
            a=np.append(a,matrix[plane_joint[i][1]][j])
            a=a.reshape(2,3)
            
            b=np.array(matrix[plane_joint[i][1]][j])
            b=np.append(b,matrix[plane_joint[i][2]][j])
            b=b.reshape(2,3)
            
            c=np.array(matrix[plane_joint[i][0]][j])
            c=np.append(c,matrix[plane_joint[i][2]][j])
            c=c.reshape(2,3)
            
            dis_a=pdist(a)
            dis_b=pdist(b)
            dis_c=pdist(c)
            
            #height using heron's formulae
            h=np.sqrt(np.square(dis_a)-np.square((np.square(dis_c)+np.square(dis_a)-np.square(dis_b))/(2*dis_c)))
            jl_distance[j][i] = np.nan_to_num(h)
    
    jj_d_col_name = []
    jj_o_col_name = []
    for i in range(numOfLine):
        jj_d_col_name.append("JJ_d: L"+str(lines[i][0])+"_L"+str(lines[i][1]))
        jj_o_col_name.append("JJ_o: L"+str(lines[i][0])+"_L"+str(lines[i][1])+" X")
        jj_o_col_name.append("JJ_o: L"+str(lines[i][0])+"_L"+str(lines[i][1])+" Y")
        jj_o_col_name.append("JJ_o: L"+str(lines[i][0])+"_L"+str(lines[i][1])+" Z")
    
    jl_d_col_name = []
    for i in range(5):
        jl_d_col_name.append("JL_d: p"+str(plane_joint[i][0])+"_p"+str(plane_joint[i][1])+"_p"+str(plane_joint[i][2]))
        
        
        
    #create dataframe
    df_jj_d = pd.DataFrame(jj_distance,columns = jj_d_col_name)
    df_jj_o = pd.DataFrame(jj_orientation,columns = jj_o_col_name)
    df_jl_d = pd.DataFrame(jl_distance,columns = jl_d_col_name)
    df_label = pd.DataFrame(np.full([frame_rate,1],label),columns=["label"])
    
    df = pd.concat([df_jj_d,df_jj_o,df_jl_d,df_label],axis=1)
    
    folder = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\Fusing_Geometric_Feature_Extracted' #change this with your system path
    root = join(folder,fileName[k])
    df.to_csv(root,index=False)
    print(fileName[k]+" success")
    
    
            
            
            
            
            
            
            
            
            
        
     