# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:23:23 2019

@author: Tony
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import shutil
import os 
import sys

from Config import *
from utility.log import Log
from function.Processing import getfirstday, fillact
from POD_DataPreparation import cut_eol_act_data, newpno_filldata_process

Trunk_folder = os.path.join(project_path , 'act_data/')
if (not os.path.exists(Trunk_folder)): 
    os.mkdir(Trunk_folder)
else:
    for folder in os.listdir(Trunk_folder):
        os.remove(Trunk_folder + folder)
    
log = Log(__name__).getlog()    

def fill_tail(key_value , asus_week_day):
    last_week = key_value['WEEK'].iloc[-1]
    last_year = key_value['YEAR'].iloc[-1]
    fixed_time = asus_week_day[(asus_week_day['year'] == last_year) & (asus_week_day['week'] == last_week)]['new_week'].iloc[0]
    future_target_time = fixed_time + safe_stock - 1

    range_weeks =  asus_week_day[(asus_week_day['new_week'] > fixed_time) & (asus_week_day['new_week'] <= future_target_time)].drop(['index','new_week'],axis=1).rename(columns = {'year':'YEAR','week':'WEEK'})
    bb = pd.merge(key_value , range_weeks , on = ['YEAR','WEEK'],how = 'outer')
    
    bb['l0name'] = bb['l0name'].fillna(key_value['l0name'].iloc[0])
    bb['PART_NO'] = bb['PART_NO'].fillna(key_value['PART_NO'].iloc[0])
    bb['act_volume'] = bb['act_volume'].fillna(0)
    return bb

if __name__ == '__main__':
    log.info("Start: ---------------get_act_data.py--------------------")

    #===========讀檔==================#
    log.info("Start: Read raw data")
    NB_activation_byweektmp = pd.read_csv(Raw_Data_path + 'NB_activation_byweektmp.csv')[['l0name', 'MODEL_NAME', 'PART_NO', 'CUSTOMER','YEAR','WEEK', 'ActWeek','act_volume']]
    pod = pd.read_csv(Sellin_path +'{}/'.format(current_week)+ 'selling_all.csv')
    eol_sol_date = pd.read_csv(Share_Data_path + 'eol_sol_date.csv')
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','date']]
    log.info("End: Read raw data")
    
    for Country in Country_List:
        log.info("Start: start {} part".format(Country))    
        
        NB_activation_byweektmp_country = NB_activation_byweektmp[NB_activation_byweektmp['l0name'] == Country]
        POD = pod[pod['l0name'] == Country]
        Model_Pno_mapping = NB_activation_byweektmp_country[['MODEL_NAME', 'PART_NO']].drop_duplicates()

        #==========前處理=================#
        log.info("Start: ready and process using table") 
        pod_pno = POD[['pno']].drop_duplicates()
        act_pno = NB_activation_byweektmp_country[['PART_NO']].drop_duplicates()
        Can_use_key = ( set(pod_pno['pno']) & set(act_pno['PART_NO']))
        Not_pod_key = set(act_pno['PART_NO']) - Can_use_key
        NB_activation_byweektmp1 = NB_activation_byweektmp_country[NB_activation_byweektmp_country['PART_NO'].isin(Can_use_key)]
        POD1 = POD[POD['pno'].isin(Can_use_key)].rename(columns = {'year':'YEAR' , 'week':'WEEK','pno':'PART_NO'})
        act_data = NB_activation_byweektmp1.groupby(['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'],as_index=False)['act_volume'].sum()
        act_data = act_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO' ,'YEAR', 'WEEK'])                
        log.info("End: ready and process using table") 
       
        #==========前處理=================#                
        log.info("Start: cut eol data (first time)")
        act_data = cut_eol_act_data(eol_sol_date, act_data)
        act_data, asus_week_day1 = newpno_filldata_process(act_data, asus_week_day, eol_sol_date)
        log.info("End:  cut eol data (first time)")
        
        #篩2013開始的key(asus_week_day不支援2013)
        log.info("Start: Screening data after 2013")
        first_have_data = act_data.groupby(['l0name', 'PART_NO'],as_index=False).first()
        Not_2013 = first_have_data[first_have_data['YEAR'] > 2013]['PART_NO'].unique()
        act_data = act_data[act_data['PART_NO'].isin(Not_2013.tolist())]
        log.info("End: Screening data after 2013")
        
        log.info("Start: fill date")                
        act_data = act_data.groupby(['l0name','MODEL_NAME','PART_NO'],as_index=False).apply(lambda x: fillact(x , asus_week_day))
        act_data = act_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO','YEAR','WEEK']).reset_index(drop=True)
        log.info("End: fill date")
        
        log.info("Start: create ylabel")
        asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','new_week']].drop_duplicates().reset_index(drop=False)     
        Ready_Data_group = act_data.groupby(['l0name','MODEL_NAME','PART_NO'])
        
        All_output = []
        for key,value in tqdm(Ready_Data_group):
            output = fill_tail(value , asus_week_day)
            All_output.append(output)
        Ready_Data1 = pd.concat(All_output).drop(['MODEL_NAME'],axis=1)
        Ready_Data1 = pd.merge(Ready_Data1 , Model_Pno_mapping ,on=['PART_NO'])
        Ready_Data1[['l0name','MODEL_NAME','PART_NO', 'YEAR', 'WEEK', 'act_volume']].to_csv(project_path + '/act_data/act_Data_'+Country+'.csv',index=False)
        del Ready_Data1,act_data
        log.info("End: create ylabel")
        
        log.info("End: End {} part".format(Country))
    log.info("End: ---------------get_act_data.py--------------------")
                
        