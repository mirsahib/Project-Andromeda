import numpy as np
from numpy import linalg as LA
from scipy.stats import skew 
import pandas as pd
import os 
import re
fileName = os.listdir()
l = len(fileName)
for k in range(0,len(fileName)):
    label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file name
    content = np.loadtxt(fileName[k])# store content of the file
    matrix = np.reshape(content,(20,int(content.shape[0]/20),4),order = "F") #reshape content in (20 joint,frame rate,xyz coordinate with confidence 													score)
        
    X = matrix[:,:,0]
    Y = matrix[:,:,1]
    Z = 400 - matrix[:,:,2]
    #left-arm
    # 0-left shoulder,7-left elbow,9-left wrist,11-left hand
    left_shoulder = np.stack((X[0,:],Y[0,:],Z[0,:]))
    left_elbow = np.stack((X[7,:],Y[7,:],Z[7,:]))
    left_wrist = np.stack((X[9,:],Y[9,:],Z[9,:]))
    left_hand = np.stack((X[11,:],Y[11,:],Z[11,:]))
    C2 = ((left_elbow+left_wrist+left_hand)/3)-left_shoulder # barycenter for left arm
    
    #right-arm
    # 1-right shoulder,8-right elbow,10-right wrist,12-right hand
    
    right_shoulder = np.stack((X[1,:],Y[1,:],Z[1,:]))
    right_elbow = np.stack((X[8,:],Y[8,:],Z[8,:]))
    right_wrist = np.stack((X[10,:],Y[10,:],Z[10,:]))
    right_hand = np.stack((X[12,:],Y[12,:],Z[12,:]))
    C3 = ((right_elbow+right_wrist+right_hand)/3)-right_shoulder  # barycenter for right arm
    
    # waist and head
    # 19-head,2-shoulder,3-spine,6-hip-center
    head = np.stack((X[19,:],Y[19,:],Z[19,:]))
    shoulder = np.stack((X[2,:],Y[2,:],Z[2,:]))
    spine = np.stack((X[3,:],Y[3,:],Z[3,:]))
    hip_center = np.stack((X[6,:],Y[6,:],Z[6,:]))
    C1 = ((head+shoulder+spine)/3)-hip_center  #barycenter for waist and head
    
    # left- leg
    # 4-left hip,13-left knee,15-left ankle,17-left foot
    left_hip = np.stack((X[4,:],Y[4,:],Z[4,:]))
    left_knee = np.stack((X[13,:],Y[13,:],Z[13,:]))
    left_ankle = np.stack((X[15,:],Y[15,:],Z[15,:]))
    left_foot = np.stack((X[17,:],Y[17,:],Z[17,:]))
    C4 = ((left_knee+left_ankle+left_foot)/3)-left_hip # barycenter for left leg
    
    # right- leg
    # 5-right hip,14-right knee,16-right-ankle,18-right-foot
    right_hip = np.stack((X[5,:],Y[5,:],Z[5,:]))
    right_knee = np.stack((X[14,:],Y[14,:],Z[14,:]))
    right_ankle = np.stack((X[16,:],Y[16,:],Z[16,:]))
    right_foot = np.stack((X[18,:],Y[18,:],Z[18,:]))
    C5 = ((right_knee+right_ankle+right_foot)/3)-right_hip # barycenter for right leg
    
    #1st level feature
    col = C1.shape[1] 
    human_matrix = np.array([C1,C2,C3,C4,C5]) #storing barycenter in one matrix
    new_human_matrix = np.empty([5,3,col])#initializing new human matrix
    label_vector = np.full([5,1],label)# initialing labal vector
    print("new_human_matrix")
    for i in range(0,5):
        parts = human_matrix[i,:,:]#store ith barycenter
        x = parts[0,:]
        y = parts[1,:]
        z = parts[2,:]
        for j in range(0,col):
            x[j] = LA.norm(x[j]-x[0]) 	#  norm(x axis-jth frame - x axis 0th value) (norm means euclidean distance)
            y[j] = LA.norm(y[j]-y[0])	#  norm(y axis-jth frame - y axis 0th value) (norm means euclidean distance)
            z[j] = LA.norm(z[j]-z[0])	#  norm(z axis-jth frame - z axis 0th value) (norm means euclidean distance)
        new_human_matrix[i,0,:] = x	#store x axis norm
        new_human_matrix[i,1,:] = y	#store y axis norm
        new_human_matrix[i,2,:] = z	#store z axis norm
    
    feature_matrix = np.empty([5,4,3])#initialize first feature matrix
    for i in range(0,5):
        feature_matrix[i,0,:] = np.amax(new_human_matrix[i,:,1:col],axis=1)- np.amin(new_human_matrix[0,:,1:col],axis=1)#range columnwise
        feature_matrix[i,1,:] = np.mean(new_human_matrix[i,:,1:col],axis=1)#mean columnwise
        feature_matrix[i,2,:] = np.var(new_human_matrix[i,:,1:col],axis=1)#variance columnwise
        feature_matrix[i,3,:] = skew(new_human_matrix[i,:,1:col],axis=1)#skewness columnwise
    #creating dataframe
    df_range = pd.DataFrame(feature_matrix[:,0,:],columns = ["RangeX","RangeY","RangeZ"])
    df_mean = pd.DataFrame(feature_matrix[:,1,:],columns = ["MeanX","MeanY","MeanZ"])
    df_variance = pd.DataFrame(feature_matrix[:,2,:],columns = ["VarX","VarY","VarZ"])
    df_skew = pd.DataFrame(feature_matrix[:,3,:],columns = ["SkewX","SkewY","SkewZ"])
    df_label = pd.DataFrame(label_vector,columns=["Label"])
    df = pd.concat([df_range,df_mean,df_variance,df_skew,df_label],axis=1)
    
    folder = r'/home/mirsahib/Downloads/OneDrive_1_29-10-2019/1st_Level_Feature_Extracted' #change this with your system path
    newFileName = fileName[k].split(".")
    root = os.path.join(folder,newFileName[0]+".csv")
    df.to_csv(root,index=False)
    print(newFileName[0]+" success")
