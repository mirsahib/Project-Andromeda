import os
import glob
import pandas as pd
# ms path
#path = r"C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\Fusing_Geometric_Feature"

path = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\2017-Mining Key Skeleton Poses'
mydir = os.chdir(path)
extension = "csv"
fn_1 = [i for i in glob.glob('*.{}'.format(extension))]
fn_1.sort()

path = r'C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\Fusing_Geometric_Feature_Extracted'
mydir = os.chdir(path)
extension = "csv"
fn_2 = [i for i in glob.glob('*.{}'.format(extension))]
fn_2.sort()

for i in fn_2:
    if i not in fn_1:
        print(i)

        

    
    



