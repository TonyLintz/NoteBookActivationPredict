# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:18:02 2019

@author: Tony
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import os
import sys
from sklearn.metrics import mean_squared_log_error
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)


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

# =============================================================================
# 計算指標
# =============================================================================
def MAPE(y_true, y_pred): 
#    y_true = [12,5,2]
#    y_pred = [16,15,11]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    Dominator = np.where(y_true!=0, y_true, 1)    
    return np.mean(np.abs((y_true - y_pred) / Dominator)) * 100

def MAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def RMSLE(y , y_hat):
    y_array = np.array(y)
    y_hat_array = np.array(y_hat)
    MSLE = mean_squared_log_error(y_array , y_hat_array)
    RMSLE = MSLE ** 0.5
    return RMSLE
    

#====================================================================================#
#                         兩邊模型計算指標                                            #
#====================================================================================#

def filter_time_origin_AE(x , Buckle2_y , Buckle2_w):
#    key = ('ID','X441UA' ,'90NB0C92-M07600')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    origin_AE = abs(x_filter['act_volume'].iloc[0:20].sum() - x_filter['predict'].iloc[0:20].sum())
    return origin_AE

def filter_time_New_AE(x , Buckle2_y , Buckle2_w):
#    key = ('ID','X441UA' ,'90NB0C92-M07600')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    new_AE = abs(x_filter['act_volume'].iloc[0:20].sum() - x_filter['POD_Predict'].iloc[0:20].sum())
    return new_AE 
 

def filter_time_origin_mape(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    origin_MAPE = MAPE(x_filter['act_volume'], x_filter['predict'])
    return origin_MAPE

def filter_time_New_mape(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    new_MAPE = MAPE(x_filter['act_volume'], x_filter['POD_Predict'])
    return new_MAPE 
    

def filter_time_origin_mae(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    origin_MAE = MAE(x_filter['act_volume'], x_filter['predict'])
    return origin_MAE

def filter_time_New_mae(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    new_MAE = MAE(x_filter['act_volume'], x_filter['POD_Predict'])
    return new_MAE 


def filter_time_origin_rmsle(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    origin_RMSLE = RMSLE(x_filter['act_volume'], x_filter['predict'])
    return origin_RMSLE

def filter_time_New_rmsle(x , Buckle2_y , Buckle2_w):
#    key = ('GB','E406MA' ,'90NB0J81-M04350')
#    x = data.groupby(['l0name', 'model', 'pno']).get_group(key)
    x_filter = x[((x['year'] == Buckle2_y) & (x['week'] <= Buckle2_w)) | (x['year'] < Buckle2_y)]
    new_RMSLE = RMSLE(x_filter['act_volume'], x_filter['POD_Predict'])
    return new_RMSLE 
   
    
#====================================================================================================#
#                                  選擇哪一邊的模型                                                   #
#====================================================================================================#
    
def choose_method(pno , check_time , All_week_indicator_df):
    #pno = '90NB0I73-M00880'

    filter_result = pd.merge(All_week_indicator_df , check_time , on = ['year','week'])
    filter_result = filter_result[filter_result['pno'] == pno]
    if len(filter_result) == 0:
        choose_result = 'origin'
    else:
        origin_win = len(np.where(filter_result['AE_Origin'] < filter_result['AE_New'])[0])*1.5 + len(np.where(filter_result['MAE_Origin'] < filter_result['MAE_New'])[0])
        new_win = len(np.where(filter_result['AE_Origin'] > filter_result['AE_New'])[0])*1.5 + len(np.where(filter_result['MAE_Origin'] > filter_result['MAE_New'])[0])
        
        if new_win >= origin_win:
                choose_result = 'New'
        else:
            choose_result = 'origin'
    
    return choose_result    

#====================================================================================================#
#                                    主程式                                                          #
#====================================================================================================#
    
if __name__ == '__main__':
    
    Folder_list = os.listdir(POD_predict_path)
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week']].drop_duplicates().reset_index(drop=False)
    NB_activation_byweektmp = pd.read_csv(Raw_Data_path + 'NB_activation_byweektmp.csv')[['l0name', 'MODEL_NAME', 'PART_NO', 'CUSTOMER','YEAR','WEEK','act_volume']]

    for folder in Folder_list:
       # folder = 'predict_result_ID'
        predict_path = POD_predict_path + folder + '/'
        predict_file = os.listdir(predict_path)
        file_list = list(filter(lambda x:  (x.endswith(".csv")) & ((str(Current_year-1) in x) | (str(Current_year) in x )) ,predict_file))
        Country = folder[-2::]
        NB_activation_byweektmp_country = NB_activation_byweektmp[NB_activation_byweektmp['l0name'] == Country]
        act_data = NB_activation_byweektmp_country.groupby(['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'],as_index=False)['act_volume'].sum()
        act_data = act_data.rename(columns = {'MODEL_NAME':'model' , 'PART_NO':'pno' , 'YEAR':'year' , 'WEEK':'week'})
        
        
        if (not os.path.exists('./Switch/')):
            os.mkdir('./Switch/')
        if (not os.path.exists('./Switch/Final_result/')):
            os.mkdir('./Switch/Final_result/')
        if (not os.path.exists('./Switch/Final_result/Final_result_'+Country+'/')):
            os.mkdir('./Switch/Final_result/Final_result_'+Country+'/')
        
        
        if (not os.path.exists('./Switch/')):
            os.mkdir('./Switch/')
        if (not os.path.exists('./Switch/Final_result_30/')):
            os.mkdir('./Switch/Final_result_30/')
        if (not os.path.exists('./Switch/Final_result_30/Final_result_'+Country+'/')):
            os.mkdir('./Switch/Final_result_30/Final_result_'+Country+'/')


        
# =============================================================================
# 計算每周預測各key指標
# =============================================================================
        All_week_indicator = []
        for file in tqdm(file_list):
            #file = '2020_w18.csv'
            
            year = int(file.split('_w')[0])
            week = int(file.split('_w')[1].split('.')[0])
            
            if (Current_week-1) <= 0:
                Buckle2_w = Current_week - 1 + 53
                Buckle2_y = Current_year - 1
            else:
                Buckle2_w = Current_week - 1 
                Buckle2_y = Current_year 
            
            if ((year == Buckle2_y) & (week > Buckle2_w)) | ((year > Buckle2_y) & (week <= Buckle2_w)):
                continue
            
            data = pd.read_csv(predict_path + file)
            data = pd.merge(data , act_data , on = ['l0name', 'model', 'pno', 'year', 'week'],how = 'left')
            data = data.drop(['act_volume_x'],axis=1).rename(columns = {"act_volume_y":'act_volume'})
            data['act_volume'] = data['act_volume'].fillna(0)
            
            origin_all_key_ae = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_origin_AE(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'AE_Origin'})
            new_all_key_ae = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_New_AE(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'AE_New'})
            
            origin_all_key_rmsle = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_origin_rmsle(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'RMSLE_Origin'})
            new_all_key_rmsle = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_New_rmsle(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'RMSLE_New'})

            origin_all_key_mae = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_origin_mae(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'MAE_Origin'})
            new_all_key_mae = data.groupby(['l0name', 'model' , 'pno']).apply(lambda x: filter_time_New_mae(x , Buckle2_y , Buckle2_w)).reset_index().rename(columns = {0:'MAE_New'})
            
            
            Merge_all_key_indicator = pd.merge(origin_all_key_ae , new_all_key_ae , on=['l0name', 'model','pno'])
            Merge_all_key_indicator = pd.merge(Merge_all_key_indicator , origin_all_key_rmsle , on =['l0name', 'model','pno'])
            Merge_all_key_indicator = pd.merge(Merge_all_key_indicator , new_all_key_rmsle , on = ['l0name', 'model','pno'])
            Merge_all_key_indicator = pd.merge(Merge_all_key_indicator , origin_all_key_mae , on =['l0name', 'model','pno'])
            Merge_all_key_indicator = pd.merge(Merge_all_key_indicator , new_all_key_mae , on = ['l0name', 'model','pno'])

           
            Merge_all_key_indicator['year'] = year
            Merge_all_key_indicator['week'] = week
            All_week_indicator.append(Merge_all_key_indicator)
            
        All_week_indicator_df = pd.concat(All_week_indicator)
        All_week_indicator_df = All_week_indicator_df.sort_values(by = ['year','week','l0name', 'model' , 'pno'])
        All_week_indicator_df = All_week_indicator_df.dropna()
        
        

# =============================================================================
# choose
# =============================================================================





        
        
        if state == 'current':
            if Current_week > 53:
                file_list = [str(Current_year)+'_w'+str(1).zfill(2)+'.csv']
            else:
                file_list = [str(Current_year)+'_w'+str(Current_week).zfill(2)+'.csv']
            
        elif state == 'back':
            predict_file = os.listdir(predict_path)
            file_list = list(filter(lambda x:  (x.endswith(".csv")) & ((str(Current_year-1) in x) | (str(Current_year) in x )) ,predict_file))
    
        try:
            file_list.remove('2018_w01.csv')
        except:
            pass        
        
        
        
        All_week_choose = []
        for file in tqdm(file_list):
            #file = '2019_w19.csv'
            year = int(file.split('_w')[0])
            week = int(file.split('_w')[1].split('.')[0])
            
            targetweek_index = asus_week_day[(asus_week_day.year == year) & (asus_week_day.week == week)].index[0]  
            check_time = asus_week_day.loc[targetweek_index-5:targetweek_index-1]
            data = pd.read_csv(predict_path + file)
            pno_list = data['pno'].unique().tolist()
            choose_list = list(map(lambda x : choose_method(x , check_time , All_week_indicator_df) ,pno_list))
            choose_result_df = pd.DataFrame()
            choose_result_df['choose'] = choose_list
            choose_result_df['pno'] = pno_list
            choose_result_df['year'] = year
            choose_result_df['week'] = week
            
            All_week_choose.append(choose_result_df)
        All_week_choose_df = pd.concat(All_week_choose)
      
        if (not os.path.exists(Raw_Data_path + 'choose/{}/'.format(Country))):
            os.mkdir(Raw_Data_path + 'choose/{}/'.format(Country))

        All_week_choose_df.to_csv(Raw_Data_path + 'choose/{}/'.format(Country)+'choose_{}.csv'.format(current_week),index=False)
            
        
#============================================開始choose預測結果=====================================================#



        All_week_choose = []
        for file in tqdm(file_list):
            #file = '2019_w9.csv'
            year = int(file.split('_w')[0])
            week = int(file.split('_w')[1].split('.')[0])
            
        
            
            #當周預測
            current_data = pd.read_csv(predict_path + file)
            current_data_group = current_data.groupby(['l0name','model','pno'])
          
            All_key_final_predict = []
            for key , value in current_data_group:
                key_choose_answer = All_week_choose_df[(All_week_choose_df['pno'] == key[2]) & (All_week_choose_df['year'] == year) & (All_week_choose_df['week'] == week)]['choose'].iloc[0]
                key_data = current_data[(current_data['l0name'] == key[0]) & (current_data['model'] == key[1]) & (current_data['pno'] == key[2])]
            
                if key_choose_answer == 'New':
                    key_data1 = key_data.drop(['predict'],axis=1).rename(columns = {'POD_Predict':'predict'})
                elif key_choose_answer == 'origin':
                    key_data1 = key_data.drop(['POD_Predict'],axis=1).rename(columns = {'predict':'predict'})
                All_key_final_predict.append(key_data1)
            All_key_final_predict_df = pd.concat(All_key_final_predict).sort_values(by = ['l0name','model','pno'])
            
           
            if len(All_key_final_predict_df) != len(current_data):
                print('#=========================raw number no match===============================#\n')
                print(file)
                break 

    
            #合併只有在origin裡的key
            
            origin_data = pd.read_csv(XGB_predict_path + file)
            origin_data_country = origin_data[origin_data['l0name'] == Country]
             
            aaaa =  pd.merge(origin_data_country , All_key_final_predict_df[['l0name', 'model', 'pno','year','week','predict']] , on = ['l0name', 'model', 'pno','year','week'] ,how = 'left')
            aaaa = aaaa.fillna('')
            origin_key = aaaa[aaaa['predict_y'] == ''][['l0name', 'model', 'pno','year','week']]
           
            origin_result = pd.merge(origin_data_country , origin_key , on=['l0name', 'model', 'pno','year','week'])
            final_predict = pd.concat([origin_result , All_key_final_predict_df[['l0name', 'model', 'pno','year','week','act_volume','predict']]])
            final_predict = final_predict.sort_values(by = ['l0name', 'model', 'pno','year','week'])
            
            if len(final_predict) != len(origin_data_country):
                print('#=========================origin predict not match New predict===============================#\n')
                print(file)
                break
            
            final_predict.to_csv('./Switch/Final_result_30/Final_result_'+Country+'/' + file ,index = False , encoding='utf-8')
            