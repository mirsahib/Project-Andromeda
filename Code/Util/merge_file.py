#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:28:28 2019

@author: mirsahib
"""

import os
import glob
import pandas as pd
# ms path
#path = r"C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\Fusing_Geometric_Feature"

path = '/home/mirsahib/Desktop/Project-Andromeda/Dataset/2017-Mining Key Skeleton Poses'
mydir = os.chdir(path)
extension = "csv"
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
all_filenames.sort()
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "MIJA_Dataset.csv", index=False, encoding='utf-8-sig')
print("success")

