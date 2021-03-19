# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:40:56 2020

@author: Tony_Tien
"""

import pandas as pd
import numpy as np  
import sys 
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

from Main_Code.Config import *


if __name__ == '__main__':
  
    #如果有需要校正齋戒月才執行，
 
    with open('./Eid_Event_Main_Code/predict_diff_eid_event.pickle', 'rb') as file:
        All_eid_diff_predict_df = pickle.load(file)
   
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','new_week']].drop_duplicates()
    predict_data = os.listdir(Event_predict_path)


    if All_eid_diff_predict_df.empty:
        
        pass
    
    else:
     
        All_eid_diff_predict_df = All_eid_diff_predict_df.rename(columns = {'MODEL_NAME':'model','PART_NO':'pno','YEAR':'year','WEEK':'week','linear_predict':'diff_predict'})
        All_eid_diff_predict_df['New_predict_value'] = All_eid_diff_predict_df['bf5_beid_mean'] + All_eid_diff_predict_df['diff_predict']
        All_eid_diff_predict_df.loc[All_eid_diff_predict_df.New_predict_value < 0,'New_predict_value'] = 0
        All_eid_diff_predict_df['New_predict_value'] = All_eid_diff_predict_df['New_predict_value'].apply(lambda x:np.round(x))
    
        
        Target_year = Current_year
        #選擇目標，回朔模式:跑當年度所有會經過當年齋戒月的預測csv  
        #當周模式:跑當周產出預測csv 
        if state == 'current':
            predict_data_year = [str(Current_year)+'_w'+str(Current_week).zfill(2)+'.csv']
        else:
            predict_data_year = list(filter(lambda x: str(Target_year) in x or str(Target_year-1) in x ,predict_data)) 

        
        All_file_condition = []
        for csv_name in predict_data_year:
            
            data = pd.read_csv(Event_predict_path + csv_name)
            year = int(csv_name.split('_w')[0])
            week = int(csv_name.split('_w')[1].split('.csv')[0])
        
            current_Yweek = str(year) + '-' + str(week).zfill(2)
            Info_data = All_eid_diff_predict_df[All_eid_diff_predict_df['Current_YWEEK'] == current_Yweek][['l0name', 'model', 'pno', 'year', 'week','New_predict_value']]
            
            add_event_df = pd.merge(data , Info_data , on = ['l0name', 'model', 'pno', 'year', 'week'],how = 'left')
            add_event_df.loc[~add_event_df.New_predict_value.isna() , 'POD_Predict'] = add_event_df.loc[~add_event_df.New_predict_value.isna(),'New_predict_value']
            add_event_df = add_event_df.drop(['New_predict_value'],axis=1)
            
           # add_event_df.to_csv(Event_predict_path + csv_name,index=False)
      
     
