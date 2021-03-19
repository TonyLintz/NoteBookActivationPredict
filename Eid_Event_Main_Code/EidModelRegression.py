# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:29:15 2020

@author: Tony_Tien
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import os
import pickle

#enviro variable
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

#import Config
from Eid_Event_Main_Code.EidUsingFuctionSet import filter_outlier



if __name__ == '__main__':
     

    
    with open('./Eid_Event_Main_Code/Eid_feature_set/Target_data.pickle', 'rb') as file:
        All_file_condition_df_Test1 = pickle.load(file)

    with open('./Eid_Event_Main_Code/Eid_feature_set/Train_data.pickle', 'rb') as file:
        All_file_condition_df_Train2 = pickle.load(file)
#===========================================================================================================================================#
    if not All_file_condition_df_Test1.empty:
        Eid_group_test = All_file_condition_df_Test1.groupby(['eid_week'])
        Eid_group_train = All_file_condition_df_Train2.groupby(['eid_week'])
        All_lifecycle_test = []
        
        for key ,value in Eid_group_test:
            value_test = Eid_group_test.get_group(key)
            value_train = Eid_group_train.get_group(key)
        
        
            test_lifecycle_group = value_test.groupby('lif_cycle_band')
            train_lifecycle_group = value_train.groupby('lif_cycle_band')
            
            lifecycle_test = []
            for key1,value1 in test_lifecycle_group:
                test_value1 = test_lifecycle_group.get_group(key1).copy()
                train_value1 = train_lifecycle_group.get_group(key1)
        
                
                p30 = np.poly1d(np.polyfit(np.array(train_value1['bf5_beid_mean']), np.array(train_value1['Y_diff']), 1))
                line_reg = p30(np.arange(min(test_value1['bf5_beid_mean']) , max(test_value1['bf5_beid_mean'])))
        
                        
                #過濾outlier
                train_lifecycle_key = train_value1.groupby(['l0name','MODEL_NAME','PART_NO'])
                can_use_train_value1 = filter_outlier(train_lifecycle_key)            

                #製作修正後回歸
                afterO_p = np.poly1d(np.polyfit(np.array(can_use_train_value1['bf5_beid_mean']), np.array(can_use_train_value1['Y_diff']), 1))
                afterO_line_reg = afterO_p(np.arange(min(test_value1['bf5_beid_mean']) , max(test_value1['bf5_beid_mean'])))
                
                
                #預測test
                predict_Y = afterO_p(test_value1['bf5_beid_mean'])
                test_value1['linear_predict'] = predict_Y
                
                lifecycle_test.append(test_value1)
            lifecycle_test_df = pd.concat(lifecycle_test)
            All_lifecycle_test.append(lifecycle_test_df)
        All_lifecycle_test_df = pd.concat(All_lifecycle_test)    
    
        
    else:
        All_lifecycle_test_df = pd.DataFrame()
    
    file = open('./Eid_Event_Main_Code/predict_diff_eid_event.pickle', 'wb')
    pickle.dump(All_lifecycle_test_df, file)
    file.close()

    
    