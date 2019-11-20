import numpy as np
from numpy import linalg as LA
from scipy.stats import skew 
import pandas as pd
#from os import listdir
#from os.path import isfile, join 
import os 
import re
#for windows
#mypath = r'C:\Users\Mir Sahib\Downloads\MSRAction3DSkeleton(20joints)'
#fileName = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#for linux
mypath = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/MSRAction3DSkeleton(20joints)'
fileName = os.listdir(mypath)

l = len(fileName)
for k in range(0,len(fileName)):
    label =int( re.findall(r'\d+', fileName[k])[0]) #extract label from file name
    #content = np.loadtxt(mypath+"\\"+fileName[k])# store content of the file
    content = np.loadtxt(mypath+"/"+fileName[k])# for linux
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
    '''
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
    '''
    
    #2nd level feature
    '''
----local origin-------
    left shoulder
    right shoulder
    left hip
    right hip
----local end-effector------
    left hand
    right hand
    left foot
    right foot
    head
----global end-effector-----
    hip-center
    '''
    #step 1 subtract all local end-effector [x,y,z] with local origin [x,y,z]
    
    l1 = left_hand-left_shoulder
    l2 = right_hand-right_shoulder
    l3 = left_foot-left_hip
    l4 = right_foot - right_hip
    l5 = head - hip_center
    g1 = left_hand-hip_center
    g2 = right_hand-hip_center
    g3 = left_foot-hip_center
    g4 = right_foot-hip_center
    g5= head-hip_center
    
    localCoordinate = np.array([l1,l2,l3,l4,l5])
    globalCoordinate = np.array([g1,g2,g3,g4,g5])
    localcol = localCoordinate[0].shape[1]
    globalcol = localCoordinate[0].shape[1]
    newlocalCoordinate = np.empty([5,3,localcol])
    newglobalCoordinate = np.empty([5,3,globalcol])
    label_vector = np.full([localcol,1],label)
    # local relative offset distance 
    for i in range(0,5):
        localX = localCoordinate[i][0]
        localY = localCoordinate[i][1]
        localZ = localCoordinate[i][0]
        for s in range(0,localcol-1):
            localX[s] = LA.norm(localX[s+1]-localX[s])
            localY[s] = LA.norm(localY[s+1]-localY[s])
            localZ[s] = LA.norm(localZ[s+1]-localZ[s])
        newlocalCoordinate[i][0] = localX
        newlocalCoordinate[i][1] = localY
        newlocalCoordinate[i][2] = localZ
    
    
    #global relative offset distance
    for i in range(0,5):
        globalX = globalCoordinate[i][0]
        globalY = globalCoordinate[i][1]
        globalZ = globalCoordinate[i][0]
        for s in range(0,globalcol-1):
            globalX[s] = LA.norm(globalX[s+1]-globalX[s])
            globalY[s] = LA.norm(globalY[s+1]-globalY[s])
            globalZ[s] = LA.norm(globalZ[s+1]-globalZ[s])
        newglobalCoordinate[i][0] = globalX
        newglobalCoordinate[i][1] = globalY
        newglobalCoordinate[i][2] = globalZ
    
    #create dataframe
    #local coordinate
    df_local_left_arm = pd.DataFrame(np.transpose(l1),columns = ["left_arm_local_X","left_arm_local_Y","left_arm_local_Z"])
    df_local_right_arm = pd.DataFrame(np.transpose(l2),columns = ["right_arm_local_X","right_arm_local_Y","right_arm_local_Z"]) 
    df_local_left_leg = pd.DataFrame(np.transpose(l3),columns = ["left_leg_local_X","left_leg_local_Y","left_leg_local_Z"])
    df_local_right_leg = pd.DataFrame(np.transpose(l4),columns = ["right_leg_local_X","right_leg_local_Y","right_leg_local_Z"])
    df_local_waist_head =pd.DataFrame(np.transpose(l5),columns = ["waist_head_local_X","waist_head_local_Y","waist_head_local_Z"])
    #relative distance local coordinate
    df_local_left_arm_dist = pd.DataFrame(np.transpose(newlocalCoordinate[0]),columns = ["left_arm_local_distX","left_arm_local__distY","left_arm_local__distZ"])
    df_local_right_arm_dist = pd.DataFrame(np.transpose(newlocalCoordinate[1]),columns = ["right_arm_local__distX","right_arm_local__distY","right_arm_local__distZ"])
    df_local_left_leg_dist = pd.DataFrame(np.transpose(newlocalCoordinate[2]),columns = ["left_leg_local__distX","left_leg_local__distY","left_leg_local__distZ"])
    df_local_right_leg_dist = pd.DataFrame(np.transpose(newlocalCoordinate[3]),columns = ["right_leg_local__distX","right_leg_local__distY","right_leg_local__distZ"])
    df_local_waist_head_dist = pd.DataFrame(np.transpose(newlocalCoordinate[4]),columns = ["waist_head_local__distX","waist_head_local__distY","waist_head_local__distZ"])
    
    #global coordinate
    df_global_left_arm = pd.DataFrame(np.transpose(g1),columns = ["left_arm_global_X","left_arm_global_Y","left_arm_global_Z"])
    df_global_right_arm = pd.DataFrame(np.transpose(g2),columns = ["right_arm_global_X","right_arm_global_Y","right_arm_global_Z"]) 
    df_global_left_leg = pd.DataFrame(np.transpose(g3),columns = ["left_leg_global_X","left_leg_global_Y","left_leg_global_Z"])
    df_global_right_leg = pd.DataFrame(np.transpose(g4),columns = ["right_leg_global_X","right_leg_global_Y","right_leg_global_Z"])
    df_global_waist_head =pd.DataFrame(np.transpose(g5),columns = ["waist_head_global_X","waist_head_global_Y","waist_head_global_Z"])
    #relative distance global coordinate
    df_global_left_arm_dist = pd.DataFrame(np.transpose(newglobalCoordinate[0]),columns = ["left_arm_global_distX","left_arm_global__distY","left_arm_global__distZ"])
    df_global_right_arm_dist = pd.DataFrame(np.transpose(newglobalCoordinate[1]),columns = ["right_arm_global__distX","right_arm_global__distY","right_arm_global__distZ"])
    df_global_left_leg_dist = pd.DataFrame(np.transpose(newglobalCoordinate[2]),columns = ["left_leg_global__distX","left_leg_global__distY","left_leg_global__distZ"])
    df_global_right_leg_dist = pd.DataFrame(np.transpose(newglobalCoordinate[3]),columns = ["right_leg_global__distX","right_leg_global__distY","right_global__leg_distZ"])
    df_global_waist_head_dist = pd.DataFrame(np.transpose(newglobalCoordinate[4]),columns = ["waist_head_global__distX","waist_head_global__distY","waist_head_global__distZ"])
    df_label = pd.DataFrame(label_vector,columns = ["label"])
    
    df = pd.concat([df_local_left_arm,df_local_right_arm,df_local_left_leg,df_local_right_leg,df_local_waist_head,
                    df_local_left_arm_dist,df_local_right_arm_dist,df_local_left_leg_dist,df_local_right_leg_dist,df_local_waist_head_dist,
                    df_global_left_arm,df_global_right_arm,df_global_left_leg,df_global_right_leg,df_global_waist_head,
                    df_global_left_arm_dist,df_global_right_arm_dist,df_global_left_leg_dist,df_global_right_leg_dist,df_global_waist_head_dist,
                    df_label
                    ],axis=1)
    folder = r'/home/mirsahib/Desktop/Project-Andromeda/Dataset/2nd_Level_Feature_Extracted' #change this with your system path
    newFileName = fileName[k].split(".")
    root = os.path.join(folder,newFileName[0]+".csv")
    df.to_csv(root,index=False)
    print(newFileName[0]+" success")

    
        
    
    
    
    #step 2 subtract all local end-effector with global end-effector