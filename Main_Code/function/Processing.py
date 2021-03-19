# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 01:19:09 2021

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

def read_data(Country):
    global NB_activation_byweektmp,pod,asus_week_day
    NB_activation_byweektmp = pd.read_csv(Raw_Data_path + 'NB_activation_byweektmp.csv')[['l0name', 'MODEL_NAME', 'PART_NO', 'CUSTOMER','YEAR','WEEK', 'ActWeek','act_volume']]
    pod = pd.read_csv(Share_Data_path + 'POD.csv')
    NB_activation_byweektmp = NB_activation_byweektmp[NB_activation_byweektmp['l0name'] == Country]
    pod = pod[pod['l0name'] == Country]
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','date']]
    return NB_activation_byweektmp,pod,asus_week_day


def dataprocessing():
    global POD1,act_data,Can_use_key
    POD = pd.merge(pod , asus_week_day , left_on = ['pod_date'] , right_on = ['date'] )
    POD = POD.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO', 'CUSTOMER' ,'year' , 'week'])

    pod_pno = POD[['PART_NO']].drop_duplicates()
    act_pno = NB_activation_byweektmp[['PART_NO']].drop_duplicates()
    
    Can_use_key = ( set(pod_pno['PART_NO']) & set(act_pno['PART_NO']) )
    Not_pod_key = set(act_pno['PART_NO']) - Can_use_key
    
    NB_activation_byweektmp1 = NB_activation_byweektmp[NB_activation_byweektmp['PART_NO'].isin(Can_use_key)]
    POD1 = POD[POD['PART_NO'].isin(Can_use_key)].rename(columns = {'year':'YEAR' , 'week':'WEEK'})
    
    act_data = NB_activation_byweektmp1.groupby(['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'],as_index=False)['act_volume'].sum()
    act_data = act_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO' ,'YEAR', 'WEEK'])
    return POD1,act_data,Can_use_key


def fill_unitil_currentweek_for_notEOL(act_data , key_data , Last_year , Last_week, asus_week_day1):    
    #key_data = not_EOL_small5_group.get_group(('FR','E402YA','90NB0MF3-M01230'))
    key_act_data = pd.merge(act_data , key_data[['l0name','MODEL_NAME','PART_NO']])
    
    earlest_year = key_act_data['YEAR'].iloc[0]
    earlest_week = key_act_data['WEEK'].iloc[0]
    
    earlest_index = asus_week_day1[(asus_week_day1['year'] == earlest_year) & (asus_week_day1['week'] == earlest_week)].index[0]
    Last_index = asus_week_day1[(asus_week_day1['year'] == Last_year) & (asus_week_day1['week'] == Last_week)].index[0]
    
    fill_data = pd.merge(key_act_data , asus_week_day1.loc[earlest_index:Last_index].rename(columns = {'year':'YEAR','week':'WEEK'}) , on = ['YEAR','WEEK'] , how='right')
    fill_data['l0name'] = fill_data['l0name'].iloc[0]
    fill_data['MODEL_NAME'] = fill_data['MODEL_NAME'].iloc[0]
    fill_data['PART_NO'] = fill_data['PART_NO'].iloc[0]
    fill_data['act_volume'] = fill_data['act_volume'].fillna(0)
    fill_data = fill_data[['YEAR','WEEK','act_volume']]
    return fill_data

def calcu_cum(POD1 , act_data):
    POD_data = POD1.groupby(['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'],as_index=False)['pcs'].sum()
    POD_data = POD_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO' ,'YEAR', 'WEEK'])
    
    All_Data = pd.merge(act_data , POD_data , on = ['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'] , how ='outer')
    All_Data['pcs'] = All_Data['pcs'].fillna(0)
    All_Data['act_volume'] = All_Data['act_volume'].fillna(0)
    All_Data = All_Data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'])
    
    return All_Data



def fillact(key_value, asus_week_day):

        First_time = key_value.iloc[0]
        Last_time = key_value.iloc[-1]
        
        asus_year_week = asus_week_day[['year', 'week']].drop_duplicates().reset_index(drop=True)
        
        start_time = asus_year_week[(asus_year_week['year'] == First_time['YEAR']) & (asus_year_week['week'] == First_time['WEEK'])].index[0]
        end_time = asus_year_week[(asus_year_week['year'] == Last_time['YEAR']) & (asus_year_week['week'] == Last_time['WEEK'])].index[0]
        week_range = asus_year_week.loc[start_time:end_time].rename(columns = {'year':'YEAR','week':'WEEK'})
        
        key_value1 = pd.merge(key_value,week_range,on=['YEAR','WEEK'],how = 'outer')
        key_value1['act_volume'] = key_value1['act_volume'].fillna(0)
        key_value1['l0name'] = key_value['l0name'].iloc[0]
        key_value1['PART_NO'] = key_value['PART_NO'].iloc[0]
        key_value1 = key_value1.sort_values(by = ['YEAR','WEEK'])
        
        if 'pcs' not in key_value.columns:
            pass
        elif 'pcs' in key_value.columns:
            key_value1['pcs'] = key_value1['pcs'].fillna(0)
        
        return key_value1
        


def getfirstday(YEAR,WEEK):          
    yearnum = str(YEAR)   #?å°å¹´ä»½
    weeknum = str(WEEK)   #?å°??
    stryearstart = yearnum +'0101'   #å½å¹´ç¬¬ä?å¤?
    yearstart = datetime.datetime.strptime(stryearstart,'%Y%m%d') #?¼å??ä¸º?¥æ??¼å?
    yearstartcalendarmsg = yearstart.isocalendar()  #å½å¹´ç¬¬ä?å¤©ç??¨ä¿¡??
    yearstartweek = yearstartcalendarmsg[1]  
    yearstartweekday = yearstartcalendarmsg[2]
    yearstartyear = yearstartcalendarmsg[0]
    if yearstartyear < int (yearnum):
        daydelat = (8-int(yearstartweekday))+(int(weeknum)-1)*7
    else :
        daydelat = (8-int(yearstartweekday))+(int(weeknum)-2)*7
     
    a = (yearstart+datetime.timedelta(days=daydelat)).date()
    return a


#切開賣後4周
def sellon_correct(All_value_df,Can_use_key):
    eol_sol_date_for_GB = pd.read_csv(Share_Data_path + 'eol_sol_date.csv')[['l0name','PART_NO','SOL_YEAR' ,'SOL_WEEK']]
    eol_sol_date_for_GB1 = eol_sol_date_for_GB[eol_sol_date_for_GB['PART_NO'].isin(Can_use_key)]
    eol_sol_date_for_GB1 = eol_sol_date_for_GB1.sort_values(by = ['SOL_YEAR' , 'SOL_WEEK'])
    
    All_value_df['first_date_week'] = All_value_df.apply(lambda x: getfirstday(x['YEAR'],x['WEEK']),axis=1)
    eol_sol_date_for_GB1['sellon_date'] = eol_sol_date_for_GB1.apply(lambda x: getfirstday(x['SOL_YEAR'],x['SOL_WEEK']),axis=1)
    eol_sol_date_for_GB1['sellon_date'] = eol_sol_date_for_GB1['sellon_date'] + datetime.timedelta(days = 35)
    All_value_df1 = pd.merge(All_value_df , eol_sol_date_for_GB1[['l0name' , 'PART_NO' , 'sellon_date']] , on = ['l0name' , 'PART_NO'])
    
    All_value_df2 = All_value_df1[All_value_df1['first_date_week'] >= All_value_df1['sellon_date']]
    
    return  All_value_df2


def EOL_cut(All_fillact_value_df , act_data):
    eol_sol_date_for_GB1 = act_data.groupby(['l0name', 'PART_NO']).apply(lambda x: x[['YEAR','WEEK']].iloc[-1]).reset_index().rename(columns = {'YEAR':'EOL_YEAR','WEEK':'EOL_WEEK'})
    All_fillact_value_df['first_date_week'] = All_fillact_value_df.apply(lambda x: getfirstday(x['YEAR'],x['WEEK']),axis=1)    
    eol_sol_date_for_GB1['elo_date'] = eol_sol_date_for_GB1.apply(lambda x: getfirstday(x['EOL_YEAR'],x['EOL_WEEK']),axis=1)
    All_fillact_value_df1 = pd.merge(All_fillact_value_df , eol_sol_date_for_GB1[['l0name', 'PART_NO' , 'elo_date']] , on = ['l0name', 'PART_NO'] , how = 'left')
    Ready_Data = All_fillact_value_df1[All_fillact_value_df1['first_date_week'] <= All_fillact_value_df1['elo_date']]
    return Ready_Data


def fill_tail(key_value , asus_week_day):
   # key_value = value
    
    last_week = key_value['WEEK'].iloc[-1]
    last_year = key_value['YEAR'].iloc[-1]
    fixed_time = asus_week_day[(asus_week_day['year'] == last_year) & (asus_week_day['week'] == last_week)]['new_week'].iloc[0]
    future_target_time = fixed_time + safe_stock - 1 

    range_weeks =  asus_week_day[(asus_week_day['new_week'] > fixed_time) & (asus_week_day['new_week'] <= future_target_time)].drop(['index','new_week'],axis=1).rename(columns = {'year':'YEAR','week':'WEEK'})
    bb = pd.merge(key_value , range_weeks , on = ['YEAR','WEEK'],how = 'outer')
    
    bb['l0name'] = bb['l0name'].fillna(key_value['l0name'].iloc[0])
    bb['PART_NO'] = bb['PART_NO'].fillna(key_value['PART_NO'].iloc[0])
    bb['stock'] = bb['stock'].fillna(key_value['stock'].iloc[-1])
    bb['act_volume'] = bb['act_volume'].fillna(0)
    bb = bb.drop(['first_date_week' , 'elo_date'],axis=1)
    return bb


def getact_insafe(value):
    #value = Ready_Data_group.get_group(('ID','90NB0FC1-M04270'))
    value['tag']= np.arange(0,len(value))
    All_safe_stock_act =[]
    All_tag = []
    for o in range(len(value)):
        if len(value.iloc[o:o+safe_stock]) == safe_stock:
            KWEEK_ACT = value['act_volume'].iloc[o:o+safe_stock].tolist()  
            
            All_safe_stock_act.extend(KWEEK_ACT)
            All_tag.extend((np.ones_like(np.arange(safe_stock)) * o).tolist() )
        else:
            break
    Y_df = pd.DataFrame() 
    Y_df['Y_lable_by_week'] = All_safe_stock_act
    Y_df['tag'] = All_tag
    
    value1 = pd.merge(value , Y_df , on = ['tag'])
    return value1


def concat_ylable_week(x,asus_week_day1):
    #x = yweek_index_df.groupby(['l0name','PART_NO','YEAR','WEEK']).get_group(('ID','90NB00K1-M04280',2014,18))
    Ylabel_asus_week = asus_week_day1.iloc[x['index'].values[0] : x['end_index'].values[0]][['year','week']]
    return Ylabel_asus_week


def get_ylabel_week(All_yweek_df,asus_week_day1):
    asus_week_day1 = asus_week_day1.reset_index()
    yweek_index_df = pd.merge(All_yweek_df , asus_week_day1 , left_on =['YEAR','WEEK'], right_on =['year','week'])
    yweek_index_df['end_index'] = yweek_index_df['index'] + safe_stock
    All_Ylabel_week = yweek_index_df.groupby(['l0name','PART_NO','YEAR','WEEK']).apply(lambda x: concat_ylable_week(x,asus_week_day1)).reset_index()
    return All_Ylabel_week