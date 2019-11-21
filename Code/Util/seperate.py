# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:14:49 2019

@author: Mir Sahib
"""
from os import listdir
from os.path import isfile, join 
import re
import shutil

path = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\1st_Level_Feature_Extracted'
newpath = r"C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\demo"
fileName = [f for f in listdir(path) if isfile(join(path, f))]
for i in range(0,len(fileName)):
    subject = int(re.findall(r'\d+', fileName[i])[1])
    instance = int(re.findall(r'\d+', fileName[i])[2])
    if  instance%2==0:
        newPath = shutil.copy(path+"\\"+fileName[i], newpath)
        print(fileName[i]+" path "+newPath)
    