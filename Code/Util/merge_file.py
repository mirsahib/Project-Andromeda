#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:28:28 2019

@author: mirsahib
"""

import os
import glob
import pandas as pd

mydir = os.chdir(r"C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\demo")
extension = "csv"
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
print("success")
