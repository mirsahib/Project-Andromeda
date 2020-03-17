#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:27:54 2020

@author: mirsahib
"""
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile,join
import re
from scipy.spatial import procrustes
from itertools import combinations


def getRefMatrix(content):
    frame_rate = content.shape[0]
    x = np.zeros([20,frame_rate])
    y = np.zeros([20,frame_rate])
    z = np.zeros([20,frame_rate])
    result = np.zeros([frame_rate,3])
    c = 0
    for i in range(20):
        joints = content[:,c:c+3]
        x[i] = joints[:,0]
        y[i] = joints[:,1]
        z[i] = joints[:,2]
        c+=3
    
    result [:,0] = np.mean(x,axis=0)
    result [:,1] = np.mean(y,axis=0)
    result [:,2] = np.mean(z,axis=0)
    
    return result
        
def procrustes_analysis(content,ref_matrix):
    frame_rate = content.shape[0]
    norm_mat = np.zeros([frame_rate,60])
    c=0
    for i in range(20):
        joint = content[:,c:c+3]
        try:
            mxt1,mxt2,disparity = procrustes(ref_matrix,joint)
            norm_mat[:,c:c+3] = mxt2
        except:
            print('ValueError')
        c+=3
    return norm_mat
    
        
    

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
        ref_matrix =getRefMatrix(content)
        norm_matrix = procrustes_analysis(content,ref_matrix)
        if norm_matrix is None:
            break
        
        # reshape
        c = 0 #internal counter
        for q in range(0,20):
            matrix[q] = norm_matrix[:,c:c+3]
            c=c+3
        
        lst = [i for i in range(20)]
        flag = 0
        for pair in combinations(lst,2):
            joint_1,joint_2 = pair
            colX = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_x'
            colY = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_y'
            colZ = 'j_'+str(joint_1)+'_j_'+str(joint_2)+'_z'
            col_name = [colX,colY,colZ]
            df = pd.DataFrame(columns=col_name)
            pair_pos = matrix[joint_1]-matrix[joint_2]
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