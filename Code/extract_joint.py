# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join

def extractJoint():
    mypath = r'C:\Users\Mir Sahib\Downloads\MSRAction3DSkeleton(20joints)'
    filename = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i in range(0,len(filename)):
        content = np.loadtxt(filename[i])
        l = int(content.shape[0]/20)
        matrix = np.reshape(content,(20,l,4),order = "F")
        X = matrix[:,:,0]
        Y = matrix[:,:,1]
        Z = 400 - matrix[:,:,2]
        
        #left arm
        left_shoulder = np.column_stack((X[0,:],Y[0,:],Z[0,:]))
        left_elbow = np.column_stack((X[7,:],Y[7,:],Z[7,:]))
        left_wrist = np.column_stack((X[9,:],Y[9,:],Z[9,:]))
        left_hand = np.column_stack((X[11,:],Y[11,:],Z[11,:]))
        
        #right arm
        right_shoulder = np.column_stack((X[1,:],Y[1,:],Z[1,:]))
        right_elbow = np.column_stack((X[8,:],Y[8,:],Z[8,:]))
        right_wrist = np.column_stack((X[10,:],Y[10,:],Z[10,:]))
        right_hand = np.column_stack((X[12,:],Y[12,:],Z[12,:]))
        
        #waist and head
        head = np.column_stack((X[19,:],Y[19,:],Z[19,:]))
        shoulder = np.column_stack((X[2,:],Y[2,:],Z[2,:]))
        spine = np.column_stack((X[3,:],Y[3,:],Z[3,:]))
        hip_center = np.column_stack((X[6,:],Y[6,:],Z[6,:]))
        
        #left leg
        left_hip = np.column_stack((X[4,:],Y[4,:],Z[4,:]))
        left_knee = np.column_stack((X[13,:],Y[13,:],Z[13,:]))
        left_ankle = np.column_stack((X[15,:],Y[15,:],Z[15,:]))
        left_foot = np.column_stack((X[17,:],Y[17,:],Z[17,:]))
        
        #right leg
        right_hip = np.column_stack((X[5,:],Y[5,:],Z[5,:]))
        right_knee = np.column_stack((X[14,:],Y[14,:],Z[14,:]))
        right_ankle = np.column_stack((X[16,:],Y[16,:],Z[16,:]))
        right_foot = np.column_stack((X[18,:],Y[18,:],Z[18,:]))
        
        df = pd.DataFrame({'left_shoulder_X':left_shoulder[:,0],'left_shoulder_Y':left_shoulder[:,1],'left_shoulder_Z':left_shoulder[:,2],
                           'left_elbow_X':left_elbow[:,0],'left_elbow_Y':left_elbow[:,1],'left_elbow_Z':left_elbow[:,2],
                           'left_wrist_X':left_wrist[:,0],'left_wrist_Y':left_wrist[:,1],'left_wrist_Z':left_wrist[:,2],
                           'left_hand_X':left_hand[:,0],'left_hand_Y':left_hand[:,1],'left_hand_Z':left_hand[:,2],        
                           'right_shoulder_X':right_shoulder[:,0],'right_shoulder_Y':right_shoulder[:,1],'right_shoulder_Z':right_shoulder[:,2],
                           'right_elbow_X':right_elbow[:,0],'right_elbow_Y':right_elbow[:,1],'right_elbow_Z':right_elbow[:,2],
                           'right_wrist_X':right_wrist[:,0],'right_wrist_Y':right_wrist[:,1],'right_wrist_Z':right_wrist[:,2],
                           'right_hand_X':right_hand[:,0],'right_hand_Y':right_hand[:,1],'right_hand_Z':right_hand[:,2],
                           'head_X':head[:,0],'head_Y':head[:,1],'head_Z':head[:,2],
                           'shoulder_X':shoulder[:,0],'shoulder_Y':shoulder[:,1],'shoulder_Z':shoulder[:,2],
                           'spine_X':spine[:,0],'Y':spine[:,1],'Z':spine[:,2],
                           'hip_center_X':hip_center[:,0],'hip_center_Y':hip_center[:,1],'hip_center_Z':hip_center[:,2],
                           'left_hip_X':left_hip[:,0],'left_hip_Y':left_hip[:,1],'left_hip_Z':left_hip[:,2],
                           'left_knee_X':left_knee[:,0],'left_knee_Y':left_knee[:,1],'left_knee_Z':left_knee[:,2],
                           'left_ankle_X':left_ankle[:,0],'left_ankle_Y':left_ankle[:,1],'left_ankle_Z':left_ankle[:,2],
                           'left_foot_X':left_foot[:,0],'left_foot_Y':left_foot[:,1],'left_foot_Z':left_foot[:,2],                       
                           'right_hip_X':right_hip[:,0],'right_hip_Y':left_hip[:,1],'right_hip_Z':right_hip[:,2],
                           'right_knee_X':right_knee[:,0],'right_knee_Y':right_knee[:,1],'right_knee_Z':right_knee[:,2],
                           'right_ankle_X':right_ankle[:,0],'right_ankle_Y':right_ankle[:,1],'right_ankle_Z':right_ankle[:,2],
                           'right_foot_X':right_foot[:,0],'right_foot_Y':right_foot[:,1],'right_foot_Z':right_foot[:,2]
                           })
        
        root = r"C:\Users\Mir Sahib\Downloads\MSRSkeletonExtracted"
        newfilename = filename[i].split(".")
        final_path = os.path.join(root,newfilename[0]+".csv")
        df.to_csv(final_path,index=False)
        print("success")

extractJoint()



    
    
