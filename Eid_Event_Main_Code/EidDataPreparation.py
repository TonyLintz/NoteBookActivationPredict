# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:55:02 2020

@author: Tony_Tien
"""

import pandas as pd
import numpy as np
import datetime
import os
import sys
from tqdm import tqdm
import pickle
import heapq

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
from Main_Code.Config import *
from Eid_Event_Main_Code.EidUsingFuctionSet import *


if __name__ == '__main__':
 
    #路徑讀取
    act_Data_ID = pd.read_csv('./act_data/act_Data_ID.csv')
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','new_week']].drop_duplicates().reset_index(drop=True)
    predict_data = os.listdir(Event_predict_path)
    #Eid_vaild = pd.read_csv('T:/All/Project_NBStockManagement/ShareData/ForTonyCumFig/FeatureEvent.csv')[['l0name', 'MODEL_NAME', 'PART_NO', 'YWeek', 'Is_Eid2']].drop_duplicates()

    
    
    #可能用到的各年齋戒月時間
    Year_Range = act_Data_ID['YEAR'].unique()
    Year_Range = Year_Range[Year_Range >= 2017]
    Eid_year_dict = get_Eid_dict(Year_Range, asus_week_day)
    
    
    #各key的bf5 mean
    act_Data_ID = act_Data_ID.groupby(['l0name', 'MODEL_NAME', 'PART_NO']).apply(lambda x:rolling_mean_and_shift(x))
    act_Data_ID['YWEEK'] = act_Data_ID.apply(lambda x: str(x['YEAR']) + '-' +str(x['WEEK']).zfill(2),axis=1)

    #actweek
    act_Data_ID = act_Data_ID.groupby(['l0name' , 'MODEL_NAME' , 'PART_NO']).apply(lambda x: get_act_week(x))
    
    #valid sku in eid
    #Eid_vaild = Eid_vaild[(Eid_vaild['Is_Eid2'] == 1) & (Eid_vaild['YWeek'].isin([Eid_year_dict[Target_year][4]]))]

    #==========================#
    #定義哪幾年齋戒月為訓練資料及目標資料，訓練資料為目標年前2年，目標資料即為指定年
    Train_year = Year_Range[Year_Range < Current_year].tolist()
    Train_year = heapq.nsmallest(2, Train_year)
    Target_year = Current_year
    
    #製作訓練集
    Each_year_train_data = []
    for year in Train_year:
         
        predict_data_year = list(filter(lambda x: str(year) in x or str(year-1) in x ,predict_data)) 
        thisyear_eid = Eid_year_dict[year]
        before_eid_week_index = asus_week_day[(asus_week_day['year'] == int(thisyear_eid[0].split('-')[0])) & (asus_week_day['week'] == int(thisyear_eid[0].split('-')[1]))].index[0] - 1
        before_eid_week = asus_week_day.loc[before_eid_week_index]
        
        before_eid_week_mean = act_Data_ID[(act_Data_ID['YEAR'] == before_eid_week['year']) & (act_Data_ID['WEEK'] == before_eid_week['week'])]

        
        All_file_condition = []
        for csv_name in tqdm(predict_data_year):
            data = pd.read_csv(Event_predict_path + csv_name)
            
            year = int(csv_name.split('_w')[0])
            week = int(csv_name.split('_w')[1].split('.csv')[0])
            current_Yweek = str(year) + '-' + str(week).zfill(2)
            current_newweek = asus_week_day[(asus_week_day['year'] == year) & (asus_week_day['week'] == week)]['new_week'].iloc[0]
            
            predict_bf5_info = data.groupby(['l0name','model','pno']).apply(lambda x:rolling_mean_and_shift_predict(x))
            data_newweek = pd.merge(predict_bf5_info , asus_week_day , on = ['year','week'])
            data_newweek['gap_week'] = data_newweek['new_week'] - current_newweek
            data_newweek['yweek'] = data_newweek.apply(lambda x: str(x['year']) + '-' + str(x['week']).zfill(2),axis=1)
            eid_event_data = data_newweek[data_newweek['yweek'].isin(thisyear_eid)]
            
            #該預測file中沒有所要的齋戒月時間，則跳過
            if eid_event_data.empty:
                continue

            
            Allkey_before_eid_bf5mean = predict_bf5_info[(predict_bf5_info['year'] == before_eid_week['year']) & (predict_bf5_info['week'] ==  before_eid_week['week'])][['l0name', 'model', 'pno','bf5_mean_predict']].rename(columns = {'bf5_mean_predict':'bf5_beid_mean'})
            
            if np.all(Allkey_before_eid_bf5mean['bf5_beid_mean'].isna()):
                 Allkey_before_eid_bf5mean = before_eid_week_mean.copy()
                 Allkey_before_eid_bf5mean = Allkey_before_eid_bf5mean.rename(columns = {'MODEL_NAME':'model','PART_NO':'pno','bf5_mean':'bf5_beid_mean'})
            eid_event_data1 = pd.merge(eid_event_data , Allkey_before_eid_bf5mean[['l0name', 'model', 'pno','bf5_beid_mean']] , on = ['l0name', 'model', 'pno'],how='left')
            
            eid_event_data1['Y_diff'] = eid_event_data1['act_volume'] - eid_event_data1['bf5_beid_mean'] 
            eid_event_data1['Current_YWEEK'] = current_Yweek
            All_file_condition.append(eid_event_data1[['l0name', 'model', 'pno', 'year', 'week' ,'Current_YWEEK','gap_week','POD_Predict', 'act_volume','shift1_mean_predict','bf5_beid_mean','Y_diff']])
            
        All_file_condition_df_Train = pd.concat(All_file_condition)
                    
                    
        #非空值部分
        No_empty_Y_data = All_file_condition_df_Train[~All_file_condition_df_Train['Y_diff'].isna()]
        
        #合併
        need_columns = ['l0name', 'model', 'pno', 'year', 'week','Current_YWEEK','gap_week','POD_Predict', 'act_volume','shift1_mean_predict','bf5_beid_mean','Y_diff']
        All_file_condition_df_Train1 = No_empty_Y_data[need_columns].copy() 
        Each_year_train_data.append(All_file_condition_df_Train1)
    Each_year_train_data_df = pd.concat(Each_year_train_data)
    
    
 #==============================================================================================#       
    
    #製作目標集      
    #選擇目標，回朔模式:跑當年度所有會經過當年齋戒月的預測csv  
    #當周模式:跑當周產出預測csv 
    if state == 'current':
        predict_data_year = [str(Current_year)+'_w'+str(Current_week).zfill(2)+'.csv']
    else:
        predict_data_year = list(filter(lambda x: str(Target_year) in x or str(Target_year-1) in x ,predict_data)) 

    thisyear_eid = Eid_year_dict[Target_year]
    before_eid_week_index = asus_week_day[(asus_week_day['year'] == int(thisyear_eid[0].split('-')[0])) & (asus_week_day['week'] == int(thisyear_eid[0].split('-')[1]))].index[0] - 1
    before_eid_week = asus_week_day.loc[before_eid_week_index]
    
    before_eid_week_mean = act_Data_ID[(act_Data_ID['YEAR'] == before_eid_week['year']) & (act_Data_ID['WEEK'] == before_eid_week['week'])]
    
    #如果Target file沒有遇到齋戒月，則直接結束
    try:
        All_file_condition = []
        for csv_name in tqdm(predict_data_year):
            data = pd.read_csv(Event_predict_path + csv_name)
            
            year = int(csv_name.split('_w')[0])
            week = int(csv_name.split('_w')[1].split('.csv')[0])
            current_Yweek = str(year) + '-' + str(week).zfill(2)
            current_newweek = asus_week_day[(asus_week_day['year'] == year) & (asus_week_day['week'] == week)]['new_week'].iloc[0]
            
            predict_bf5_info = data.groupby(['l0name','model','pno']).apply(lambda x:rolling_mean_and_shift_predict(x))
            data_newweek = pd.merge(predict_bf5_info , asus_week_day , on = ['year','week'])
            data_newweek['gap_week'] = data_newweek['new_week'] - current_newweek
            data_newweek['yweek'] = data_newweek.apply(lambda x: str(x['year']) + '-' + str(x['week']).zfill(2),axis=1)
            eid_event_data = data_newweek[data_newweek['yweek'].isin(thisyear_eid)]
            
            #該預測file中沒有所要的齋戒月時間，則跳過
            if eid_event_data.empty:
                continue
    

            
            Allkey_before_eid_bf5mean = data[(data['year'] == before_eid_week['year']) & (data['week'] ==  before_eid_week['week'])][['l0name', 'model', 'pno','POD_Predict']].rename(columns = {'POD_Predict':'bf5_beid_mean'})
            
            if np.all(Allkey_before_eid_bf5mean['bf5_beid_mean'].isna()):
                 Allkey_before_eid_bf5mean = before_eid_week_mean.copy()
                 Allkey_before_eid_bf5mean = Allkey_before_eid_bf5mean.rename(columns = {'MODEL_NAME':'model','PART_NO':'pno','bf5_mean':'bf5_beid_mean'})
            eid_event_data1 = pd.merge(eid_event_data , Allkey_before_eid_bf5mean[['l0name', 'model', 'pno','bf5_beid_mean']] , on = ['l0name', 'model', 'pno'],how='left')
            
            eid_event_data1['Y_diff'] = eid_event_data1['act_volume'] - eid_event_data1['bf5_beid_mean'] 
            eid_event_data1['Current_YWEEK'] = current_Yweek
            All_file_condition.append(eid_event_data1[['l0name', 'model', 'pno', 'year', 'week' ,'Current_YWEEK','gap_week','POD_Predict', 'act_volume','shift1_mean_predict','bf5_beid_mean','Y_diff']])
            
        All_file_condition_df_Test = pd.concat(All_file_condition)
        
        #非空值部分
        No_empty_Y_data = All_file_condition_df_Test[~All_file_condition_df_Test['Y_diff'].isna()]
        
        #合併
        need_columns = ['l0name', 'model', 'pno', 'year', 'week','Current_YWEEK','gap_week','POD_Predict', 'act_volume','shift1_mean_predict','bf5_beid_mean','Y_diff']
        All_file_condition_df_Test1 = No_empty_Y_data[need_columns].copy() 
    #====================================加入lifecycle group與eidweek===========================================================# 
        
        
        All_file_condition_df_Train = Each_year_train_data_df.rename(columns = {'model':'MODEL_NAME','pno':'PART_NO','year':'YEAR','week':'WEEK'})
        All_file_condition_df_Test1 = All_file_condition_df_Test1.rename(columns = {'model':'MODEL_NAME','pno':'PART_NO','year':'YEAR','week':'WEEK'})
        
        
        #訓練集與目標集加入Eidweek
        for year in Train_year:
            All_file_condition_df_Train.loc[(All_file_condition_df_Train.YEAR == year) & (All_file_condition_df_Train.WEEK == int(Eid_year_dict[year][0].split('-')[1])),'eid_week'] = 1
            All_file_condition_df_Train.loc[(All_file_condition_df_Train.YEAR == year) & (All_file_condition_df_Train.WEEK == int(Eid_year_dict[year][1].split('-')[1])),'eid_week'] = 2
            All_file_condition_df_Train.loc[(All_file_condition_df_Train.YEAR == year) & (All_file_condition_df_Train.WEEK == int(Eid_year_dict[year][2].split('-')[1])),'eid_week'] = 3
            All_file_condition_df_Train.loc[(All_file_condition_df_Train.YEAR == year) & (All_file_condition_df_Train.WEEK == int(Eid_year_dict[year][3].split('-')[1])),'eid_week'] = 4
#            All_file_condition_df_Train.loc[(All_file_condition_df_Train.YEAR == year) & (All_file_condition_df_Train.WEEK == int(Eid_year_dict[year][4].split('-')[1])),'eid_week'] = 5 
        
    
        All_file_condition_df_Test1.loc[(All_file_condition_df_Test1.YEAR == Target_year) & (All_file_condition_df_Test1.WEEK == int(Eid_year_dict[Target_year][0].split('-')[1])),'eid_week'] = 1
        All_file_condition_df_Test1.loc[(All_file_condition_df_Test1.YEAR == Target_year) & (All_file_condition_df_Test1.WEEK == int(Eid_year_dict[Target_year][1].split('-')[1])),'eid_week'] = 2
        All_file_condition_df_Test1.loc[(All_file_condition_df_Test1.YEAR == Target_year) & (All_file_condition_df_Test1.WEEK == int(Eid_year_dict[Target_year][2].split('-')[1])),'eid_week'] = 3
        All_file_condition_df_Test1.loc[(All_file_condition_df_Test1.YEAR == Target_year) & (All_file_condition_df_Test1.WEEK == int(Eid_year_dict[Target_year][3].split('-')[1])),'eid_week'] = 4
#        All_file_condition_df_Test1.loc[(All_file_condition_df_Test1.YEAR == Target_year) & (All_file_condition_df_Test1.WEEK == int(Eid_year_dict[Target_year][4].split('-')[1])),'eid_week'] = 5
        
        
        All_file_condition_df_Train2 = pd.merge(All_file_condition_df_Train ,  act_Data_ID[['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','act_week']] , on = ['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK'] )
        All_file_condition_df_Test1 = pd.merge(All_file_condition_df_Test1 ,  act_Data_ID[['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','act_week']] , on = ['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK'])
        
        #訓練集與目標集加入life cycle group
        All_file_condition_df_Train2 = get_life_cycle_band(All_file_condition_df_Train2)
        All_file_condition_df_Test1 = get_life_cycle_band(All_file_condition_df_Test1)
        
        #篩出在齋戒月有效下降的SKU
        #All_file_condition_df_Test1 = All_file_condition_df_Test1[All_file_condition_df_Test1['PART_NO'].isin(Eid_vaild['PART_NO'].tolist())]

        
        
    except:
        All_file_condition_df_Test1 = pd.DataFrame()
        All_file_condition_df_Train2 = pd.DataFrame()
                #===============================================================================================#   
    #存取訓練集及目標集
    if (not os.path.exists('./Eid_Event_Main_Code/Eid_feature_set/')): 
        os.mkdir('./Eid_Event_Main_Code/Eid_feature_set/')
        
    
    file = open('./Eid_Event_Main_Code/Eid_feature_set/Train_data.pickle', 'wb')
    pickle.dump(All_file_condition_df_Train2, file)
    file.close()
    
    file = open('./Eid_Event_Main_Code/Eid_feature_set/Target_data.pickle', 'wb')
    pickle.dump(All_file_condition_df_Test1, file)
    file.close()

        
        
    
    


    
    
    
    
    
    
    
    
    