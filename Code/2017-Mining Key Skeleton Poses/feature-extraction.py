#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:31:59 2020

@author: mirsahib
"""
import numpy as np
from numpy import linalg as LA
import pandas as pd
from os import listdir
from os.path import isfile, join 
import re
from itertools import combinations
from sklearn.preprocessing import StandardScaler


def main():
    mypath = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/MSRAction3D_Skeleton_Joint_Extracted'
    fileName = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    fileName.sort()
    l = len(fileName)
    for k in range(0,l):
        label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file  name
        content = np.loadtxt(open(mypath+"/"+fileName[k], "rb"), delimiter=",", skiprows=1)
        matrix = np.zeros((20,content.shape[0],3))
        frame_rate = content.shape[0]
        # reshape
        c = 0 #internal counter
        for q in range(0,20):
            matrix[q] = content[:,c:c+3]
            c=c+3
        
        lst = [i for i in range(20)]
        
        
        #pairwise relative position
        flag = 0
        for pair in combinations(lst,2):
            joint_1,joint_2 = pair
            colX = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_x'
            colY = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_y'
            colZ = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_z'
            col_name = [colX,colY,colZ]
            df = pd.DataFrame(columns=col_name)
            pair_pos = StandardScaler().fit_transform(matrix[joint_1]-matrix[joint_2])
            df[colX] = pair_pos[:,0]
            df[colY] = pair_pos[:,1]
            df[colZ] = pair_pos[:,2]
            if flag!=0:
                df = pd.concat([oldDf,df],axis=1)
                oldDf = df
            else:
                oldDf = df
                flag=1
                
        jointDf = pd.read_csv(mypath+"/"+fileName[k])    
        #create dataframe
        df_label = pd.DataFrame(np.full([frame_rate,1],label),columns=["label"])
        df = pd.concat([df,jointDf,df_label],axis=1)
        folder = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/2017-Mining Key Skeleton Poses' #change this with your system path
        root = join(folder,fileName[k])
        df.to_csv(root,index=False)
        print(fileName[k]+" success")
        
        

main()