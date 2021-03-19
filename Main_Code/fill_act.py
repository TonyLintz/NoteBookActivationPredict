# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:31:27 2019

@author: Tony_Tien
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import os
import sys

args = sys.argv
if (len(args) > 1):
   print (("workspace is :", args[1]))
   os.chdir(args[1])
   sys.path.append(args[1])  
else: 
   strr = "\\"
   seq = os.getcwd().split(strr)[:-1] 
   workspace = strr.join(seq)
   os.chdir(workspace)
   sys.path.append(workspace)  
   print("workspace is :", workspace)



from Main_Code.Config import *



if __name__ == '__main__':
    
    
     Origin_predict_list = os.listdir(origin_path)
     Origin_predict_list1 = list(filter(lambda x:  x.endswith(".csv") ,Origin_predict_list))
     
     POD_predict_list = os.listdir(output_path)
     POD_predict_list1 = list(filter(lambda x:  x.endswith(".csv") ,POD_predict_list))
    
    
     for file in tqdm(POD_predict_list1):
         
         origin_predict = pd.read_csv(origin_path + file)
         origin_predict = origin_predict.sort_values(by = ['l0name', 'model', 'pno', 'year', 'week']).reset_index(drop=True)
         switch_predict = pd.read_csv(output_path + file)
         switch_predict = switch_predict.sort_values(by = ['l0name', 'model', 'pno', 'year', 'week']).reset_index(drop=True)
         
  
         if len(origin_predict) != len(switch_predict):
                  break
         
         switch_predict['act_volume'] = origin_predict['act_volume']
         
         if any(origin_predict['act_volume'] != switch_predict['act_volume']) :
                 break
         
         switch_predict.to_csv(output_path + file,index=False)