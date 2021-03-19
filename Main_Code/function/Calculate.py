# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 01:22:34 2021

@author: Tony_Tien
"""
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from matplotlib import pyplot as plt
from scipy.integrate import simps

#==========Config=================#
global safe_stock 
safe_stock = 20
from Config import *
#=================================#

def add_act_week(key_value):
    key_value['act_week'] = np.arange(6,len(key_value)+6)
    return key_value
    

def add_future_stock(All_feature_df1,All_add_future_sellin):
    Not_NAN = (All_feature_df1.dropna()).index.tolist()
    
    key_add_future_sellin = All_add_future_sellin[All_add_future_sellin['PART_NO'] == All_feature_df1['PART_NO'].iloc[0]]
    All_futurestock_list= []
    for i in Not_NAN:
        add_future_sellin = key_add_future_sellin.iloc[i:i+safe_stock].copy()
        add_future_sellin['pcs_y'] = add_future_sellin['pcs_y'].fillna(0)
        piece_cumsumpcs = add_future_sellin['pcs_y'].cumsum()
        
        All_futurestock_list.append( (piece_cumsumpcs + All_feature_df1['before_1week_stock'].iloc[i]).tolist())
    na_list = np.arange(len(All_feature_df1)-len(Not_NAN))*np.nan
    All_futurestock_list.extend(na_list)
    return All_futurestock_list


def get_sku_sellin_length_table(All_feature_df1,All_add_future_sellin): 
#    All_add_future_sellin = future_sellin.copy()
#    All_feature_df1 = Ready_Data1.groupby(['l0name','PART_NO']).get_group(('ID','90NB0H41-M00590'))

    Not_NAN = (All_feature_df1.dropna()).index.tolist()
    
    key_add_future_sellin = All_add_future_sellin[All_add_future_sellin['PART_NO'] == All_feature_df1['PART_NO'].iloc[0]]
    All_SKU_series_ = []
    for i in Not_NAN:
        add_future_sellin = key_add_future_sellin.iloc[i:i+safe_stock].copy()
        add_future_sellin['pcs_y'] = add_future_sellin['pcs_y'].fillna(0)
        
        add_future_sellin['sellin_freq'] = len(add_future_sellin[add_future_sellin['pcs_y'] != 0]) / safe_stock
        All_SKU_series_.append(add_future_sellin[['l0name', 'PART_NO', 'YEAR', 'WEEK','sellin_freq']].iloc[[0]])
    All_SKU_series_df = pd.concat(All_SKU_series_)

    return All_SKU_series_df[['YEAR', 'WEEK','sellin_freq']]


#平均特徵
def get_mean_feature(value):
    Range = 5

    if len(value) <= Range :
#        value = value.drop(['until_act'],axis=1)
        return value
    
    Real_All_first_4_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
    Real_All_until_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
    first_4w_na = (np.ones_like(np.arange(Range))*np.nan).tolist()

    
    All_until_act_mean_value = []
    All_until_act_std_value = []
    for i in range(Range , len(value)):
        until_act_mean_value = value['act_volume'].rolling(window=i).mean().tolist()[i-1]
        until_act_std_value = value['act_volume'].rolling(window=i).apply(lambda x: np.std(x)).tolist()[i-1]
        
        All_until_act_mean_value.append(until_act_mean_value)
        All_until_act_std_value.append(until_act_std_value)
        
    mean_rolling = value['act_volume'].rolling(window=Range).mean()[0:-1].tolist()
    mean_rolling.insert(0 , np.nan)
    
    std_rolling = value['act_volume'].rolling(window=Range).apply(lambda x: np.std(x))[0:-1].tolist()
    std_rolling.insert(0 , np.nan)
    
    value['first_4_act_mean'] = mean_rolling
    value['first_4_act_std'] = std_rolling
   
    Real_All_first_4_act.extend(All_until_act_mean_value)
    value['until_act_mean'] = Real_All_first_4_act
    
    Real_All_until_act.extend(All_until_act_std_value)
    value['until_act_std'] = Real_All_until_act
    
    last_4w = value['act_volume'].shift(4)
    last_3w = value['act_volume'].shift(3)
    last_2w = value['act_volume'].shift(2)
    last_1w = value['act_volume'].shift(1)
    last_0w = value['act_volume'].shift(0)
    a = pd.concat([last_4w ,last_3w,last_2w,last_1w,last_0w],axis=1).dropna()[0:-1]
    a_list =  a.values.tolist()
    first_4w_na.extend(a_list)
    value['first_4_act'] = first_4w_na
    return value


def cal_slope_change(p_list):
    All_slope_change = []
    for i in range(len(p_list)-1):
        if type(p_list) == list:
            if p_list[i] != 0:
                slope_change = (p_list[i+1] - p_list[i])/p_list[i]
            else:
                slope_change = (p_list[i+1] - p_list[i])/1
              
        else:
            slope_change = np.nan
        All_slope_change.append(slope_change)
    return All_slope_change


#斜率特徵
def get_slope_feature(value):
    Range = 5
    value['first_4_act'] = np.nan
    value['until_act'] = np.nan
    if len(value) <= Range :
        value = value.drop(['first_4_act'],axis=1)
        value = value.drop(['until_act'],axis=1)
        return value
    Real_All_first_4_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
    Real_All_until_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
  
    All_first_4_act =[]
    All_until_act =[]
    for o in range(Range,len(value)):    

            FIRST4_ACT = value['act_volume'].iloc[o-Range:o].tolist()  
            UNTIL_ACT = value['act_volume'].iloc[0:o].tolist()  
            All_first_4_act.append(FIRST4_ACT)
            All_until_act.append(UNTIL_ACT)
    Real_All_first_4_act.extend(All_first_4_act)
    Real_All_until_act.extend(All_until_act)
    value['first_4_act'] = Real_All_first_4_act
            
    value['first_4_act_slope'] = value['first_4_act'].iloc[5::].apply(lambda a : cal_slope_change(a))
    value['change_slope_mean'] = value['first_4_act_slope'].apply(lambda x: np.mean(x))
    value = value.drop(['first_4_act'],axis=1)
    value = value.drop(['until_act'],axis=1)
    return value


def feature_set_clean(All_new_value_df2):
   
    first_4_act_slope_df = pd.DataFrame(All_new_value_df2['first_4_act_slope'].values.tolist() , columns = ['First_slope','Second_slope','Third_slope','Fourth_slope'])
    first_4_act_df = pd.DataFrame(All_new_value_df2['first_4_act'].values.tolist() , columns = ['First_act','Second_act','Third_act','Fourth_act','Fiveth_act'])
    All_new_value_df3 = pd.concat([All_new_value_df2 , first_4_act_slope_df , first_4_act_df],axis=1)

    columns_sort = ['l0name', 'PART_NO', 'YEAR', 'WEEK','sellin_freq' ,'first_4_act_mean','until_act_mean','first_4_act_std','until_act_std' , 'First_act','Second_act','Third_act' ,'Fourth_act','Fiveth_act','change_slope_mean','First_slope','Second_slope','Third_slope','Fourth_slope','before_1week_stock','W1_stock','W2_stock', 'W3_stock', 'W4_stock', 'W5_stock','W6_stock', 'W7_stock', 'W8_stock', 'W9_stock', 'W10_stock','W11_stock', 'W12_stock', 'W13_stock', 'W14_stock', 'W15_stock','W16_stock', 'W17_stock', 'W18_stock', 'W19_stock', 'W20_stock','act_week']
 
    All_new_value_df3 = All_new_value_df3[columns_sort]
    
    Feature_set = All_new_value_df3.copy()
    Feature_set = Feature_set.sort_values(by = ['l0name', 'PART_NO', 'YEAR', 'WEEK'])

    return Feature_set


def cal_slope_change(p_list):
    All_slope_change = []
    for i in range(len(p_list)-1):
        if type(p_list) == list:
            if p_list[i] != 0:
                slope_change = (p_list[i+1] - p_list[i])/p_list[i]
            else:
                slope_change = (p_list[i+1] - p_list[i])/1
              
        else:
            slope_change = np.nan
        All_slope_change.append(slope_change)
    return All_slope_change


def Model_processing(act_model,asus_week_day , Current_year , Current_week):
                
        #=========#

        Current_Model_act = act_model[(act_model['YEAR'] == Current_year) & (act_model['WEEK'] == Current_week)]
        Current_Model_act_group = Current_Model_act.groupby(['l0name','MODEL_NAME'])
        
        act_model_back = act_model.drop(Current_Model_act.index)
        
        
        Current_first4W_index =  asus_week_day[(asus_week_day['year'] == Current_year) & (asus_week_day['week'] == Current_week)].index[0]
        Current_first4W = asus_week_day.loc[Current_first4W_index - 4 : Current_first4W_index-1]
        Current_first4W = Current_first4W.rename(columns = {'year':'YEAR' , 'week': 'WEEK'})
        ss = pd.merge(act_model , Current_first4W , on = ['YEAR', 'WEEK'])
        
        
        All_change_currentweek = []
        for key,value in Current_Model_act_group:

             ssa = ss[(ss['l0name'] == key[0]) & (ss['MODEL_NAME'] == key[1])]
             mean_act_first_4 =  np.round(ssa['act_volume'].mean())
             value['act_volume'] = mean_act_first_4
             All_change_currentweek.append(value)
        All_change_currentweek_df = pd.concat(All_change_currentweek)
        
        act_model = pd.concat([act_model_back , All_change_currentweek_df]).sort_values(by = ['l0name','MODEL_NAME','YEAR','WEEK'])
        
        return act_model
    
    
def get_model_mean_feature(value):
    Range = 5

    if len(value) <= Range :
#        value = value.drop(['until_act'],axis=1)
        return value
    
    Real_All_first_4_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
    Real_All_until_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
    first_4w_na = (np.ones_like(np.arange(Range))*np.nan).tolist()

    
    All_until_act_mean_value = []
    All_until_act_std_value = []
    for i in range(Range , len(value)):
        until_act_mean_value = value['act_volume'].rolling(window=i).median().tolist()[i-1]
        until_act_std_value = value['act_volume'].rolling(window=i).apply(lambda x: np.std(x)).tolist()[i-1]
        
        All_until_act_mean_value.append(until_act_mean_value)
        All_until_act_std_value.append(until_act_std_value)
        
    mean_rolling = value['act_volume'].rolling(window=Range).mean()[0:-1].tolist()
    mean_rolling.insert(0 , np.nan)
    
    std_rolling = value['act_volume'].rolling(window=Range).apply(lambda x: np.std(x))[0:-1].tolist()
    std_rolling.insert(0 , np.nan)
    
    value['first_4_act_mean'] = mean_rolling
    value['first_4_act_std'] = std_rolling
    
    
      
    Real_All_first_4_act.extend(All_until_act_mean_value)
    value['until_act_mean'] = Real_All_first_4_act
    
    Real_All_until_act.extend(All_until_act_std_value)
    value['until_act_std'] = Real_All_until_act
    
    last_4w = value['act_volume'].shift(4)
    last_3w = value['act_volume'].shift(3)
    last_2w = value['act_volume'].shift(2)
    last_1w = value['act_volume'].shift(1)
    last_0w = value['act_volume'].shift(0)
    a = pd.concat([last_4w ,last_3w,last_2w,last_1w,last_0w],axis=1).dropna()[0:-1]
    a_list =  a.values.tolist()
    first_4w_na.extend(a_list)
    value['first_4_act'] = first_4w_na
    #value = value[['YEAR', 'WEEK', 'act_volume','first_4_act_mean', 'first_4_act_std', 'until_act_mean','until_act_std', 'first_4_act']]
    
    return value


def get_model_slope_feature(value):
        Range = 5
        value['first_4_act'] = np.nan
        value['until_act'] = np.nan
        if len(value) <= Range :
            value = value.drop(['first_4_act'],axis=1)
            value = value.drop(['until_act'],axis=1)
            return value
        Real_All_first_4_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
        Real_All_until_act = (np.ones_like(np.arange(Range))*np.nan).tolist()
        All_first_4_act =[]
        All_until_act =[]
        for o in range(Range,len(value)):       
                FIRST4_ACT = value['act_volume'].iloc[o-Range:o].tolist()  
                UNTIL_ACT = value['act_volume'].iloc[0:o].tolist()  
                All_first_4_act.append(FIRST4_ACT)
                All_until_act.append(UNTIL_ACT)
    
           
            
        
        Real_All_first_4_act.extend(All_first_4_act)
        Real_All_until_act.extend(All_until_act)
        value['first_4_act'] = Real_All_first_4_act
     
        def cal_slope_change(p_list):
            All_slope_change = []
            for i in range(len(p_list)-1):
                if type(p_list) == list:
                    if p_list[i] != 0:
                        slope_change = (p_list[i+1] - p_list[i])/p_list[i]
                    else:
                        slope_change = (p_list[i+1] - p_list[i])/1
                      
                else:
                    slope_change = np.nan
                All_slope_change.append(slope_change)
            return All_slope_change
                
                
        value['first_4_act_slope'] = value['first_4_act'].iloc[5::].apply(lambda a : cal_slope_change(a))
        value['change_slope_mean'] = value['first_4_act_slope'].apply(lambda x: np.mean(x))
        value = value.drop(['first_4_act'],axis=1)
        value = value.drop(['until_act'],axis=1)
        return value


def kk(x , asus_week_day , safe_stock,year_week_act):
        #x = Feature_set.iloc[0]
        specify = asus_week_day[(asus_week_day['year'] == x['YEAR']) & (asus_week_day['week'] == x['WEEK'])].index[0]
        end = specify + safe_stock - 1
        future14w = asus_week_day.loc[specify:end]['week'].values.tolist()
        
        x_list = pd.DataFrame(future14w,columns = ['WEEK'])
        y_list = pd.merge(x_list , year_week_act[['WEEK','normalize_act']] , on =['WEEK'])['normalize_act'].values.tolist()
        return y_list
    

def get_area_feature(value ,act_data , GB_model_act):
    #value = Feature_set.groupby(['l0name', 'MODEL_NAME', 'PART_NO'],as_index=False).get_group(('ID','X441MA','90NB0H41-M00590'))

                ModelName = value['MODEL_NAME'].iloc[0]
                Pno = value['PART_NO'].iloc[0]
                Country = value['l0name'].iloc[0]
                
                pno_value1 = act_data[(act_data['PART_NO'] == Pno) & (act_data['l0name'] == Country)]
                model_value1 = GB_model_act[(GB_model_act['MODEL_NAME'] == ModelName) & (GB_model_act['l0name'] == Country)]
                
                the_same_period = pd.merge(pno_value1 , model_value1[['l0name', 'MODEL_NAME', 'YEAR', 'WEEK', 'act_volume']] , on = ['l0name','MODEL_NAME','YEAR','WEEK']).rename(columns = {'act_volume_x':'pno_act','act_volume_y':'model_act'})
                
                Range = 5
            
                #All_until_act_area = (np.ones_like(np.arange(Range))*-1).tolist()
                All_until_act_area = []
                for o in range(Range,len(the_same_period)):       
                       
                        until_pno_act = the_same_period['pno_act'].iloc[0:o].tolist()  
                        until_model_act = the_same_period['model_act'].iloc[0:o].tolist()
                        model_area = simps(until_model_act, dx=1)
                
                        
                        if model_area == 0:
                            area_ratio = 0
                        else:
            
                            pno_area = simps(until_pno_act, dx=1)
                            area_ratio = pno_area / model_area
                        
                        All_until_act_area.append(area_ratio)
               
                return All_until_act_area
            
            
def fill_area_ratio(value , Area_Ratio_Chart):
    key = (value['l0name'].iloc[0] , value['MODEL_NAME'].iloc[0] , value['PART_NO'].iloc[0])
    specifykey_arearatio = Area_Ratio_Chart[(Area_Ratio_Chart['l0name'] == key[0]) & (Area_Ratio_Chart['PART_NO'] == key[2])]['area_ratio_list'].iloc[0]
    value['area_ratio'] = specifykey_arearatio
    return value 