# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:22:48 2020

@author: Mir Sahib
"""
import numpy as np
from numpy import linalg as LA
import pandas as pd
from os import listdir
from os.path import isfile, join 
import re
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import math

def orientation_of_joint(jc,hc,matrix,frame_rate):
    ij = np.zeros((11,frame_rate,3))#interested joint
    for i in range(len(jc)):
        ij[i] = matrix[jc[i]]-matrix[hc]
        ij[i] = StandardScaler().fit_transform(ij[i]) #normalize the data

    return ij

def cart2sph(x, y, z,frame_rate):
    data = np.zeros([frame_rate,2])
    hxy = np.hypot(x, y)
    #ra = np.hypot(hxy, z)
    #el = np.arctan2(z, hxy)
    #az = np.arctan2(y, x)
    data[:,0] = np.around(np.arctan2(y, x) * (180/math.pi))
    data[:,1] = np.around(np.arctan2(z, hxy) * (180/math.pi))
    #data[:,2] = np.hypot(hxy, z)
    return data

#def cartisian_to_spherical(jc,interested_joint):
#    for i in range(len(jc)):
#        for j in range(len(interested_joint[jc[i]])):
#            interested_joint[jc[i]][j] = cart2sph(interested_joint[jc[i]][j][0],interested_joint[jc[i]][j][1],interested_joint[jc[i]][j][2])
def cartisian_to_spherical(jc,interested_joint):
    sphericalJoint = np.zeros([11,interested_joint.shape[1],2])
    for i in range(len(jc)):
        sphericalJoint[i] = cart2sph(interested_joint[i][:,0],interested_joint[i][:,1],interested_joint[i][:,2],interested_joint[i].shape[0])
    return sphericalJoint
            
        

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
        joints_coordinate = np.array([8,9,10,11,12,13,14,15,18,19,20])
        hip_coordinate = 6
        joints_coordinate = joints_coordinate-1
        interested_joint = orientation_of_joint(joints_coordinate,hip_coordinate,matrix,frame_rate)
        spherical_coordinate = cartisian_to_spherical(joints_coordinate,interested_joint)
        flag = 0
        for i in range(len(joints_coordinate)):
            col_name = []
            col_name.append("Joint_"+str(joints_coordinate[i])+"_az")
            col_name.append("Joint_"+str(joints_coordinate[i])+"_el")
            #col_name.append("Joint_"+str(joints_coordinate[i])+"_r")
            df = pd.DataFrame(spherical_coordinate[i],columns = col_name)
            if flag!=0:
                df = pd.concat([olddf,df],axis=1)
                olddf = df
            else:
                olddf = df
                flag=1
                
            
        #create dataframe
        df_label = pd.DataFrame(np.full([frame_rate,1],label),columns=["label"])
        df = pd.concat([df,df_label],axis=1)
        folder = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/MIJA_Round' #change this with your system path
        root = join(folder,fileName[k])
        df.to_csv(root,index=False)
        print(fileName[k]+" success")
        
        

main()