# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:19:57 2020

@author: Tony_Tien
"""


import numpy as np
from  Eid_Event_Main_Code import IDEvent
import datetime
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#獲得各年齋戒月周數
def get_Eid_dict(Year_Range , asus_week_day):
    Eid_year_dict = {}
    for Year in Year_Range:
        obj = IDEvent.Indonesia()
        obj._populate(Year)
        event = [i for i in obj.items()]
        Eid_event = ["Eid al-Fitr" in i for i in event]
        Eid_date = [i for indx,i in enumerate(event) if Eid_event[indx] == True][0][0] - datetime.timedelta(1)
        Eid_start = Eid_date - datetime.timedelta(21)
        
        Eid_start_week = Eid_start.isocalendar()[1]
        Eid_end_week = Eid_date.isocalendar()[1]
        
        start_year = Eid_start.year
        end_year = Eid_date.year        
        
        start_index = asus_week_day[(asus_week_day['year'] == start_year) & (asus_week_day['week'] == Eid_start_week)].index[0]
        end_index = asus_week_day[(asus_week_day['year'] == end_year) & (asus_week_day['week'] == Eid_end_week)].index[0]
        Eid_all_week = asus_week_day.loc[start_index:end_index]
        Eid_year_dict[Year] = Eid_all_week.apply(lambda x: str(x['year']) + '-' + str(x['week']).zfill(2),axis=1).values.tolist()
    return Eid_year_dict



#齋戒月周數label
def get_eid_week(x,year):
    x['eid_week'] = np.nan
    if year == 2017: 
       x.loc[x.WEEK == 22,'eid_week'] = 1
       x.loc[x.WEEK == 23,'eid_week'] = 2
       x.loc[x.WEEK == 24,'eid_week'] = 3
       x.loc[x.WEEK == 25,'eid_week'] = 4
       x.loc[x.WEEK == 26,'eid_week'] = 5
    elif year == 2018:
       x.loc[x.WEEK == 20,'eid_week'] = 1
       x.loc[x.WEEK == 21,'eid_week'] = 2
       x.loc[x.WEEK == 22,'eid_week'] = 3
       x.loc[x.WEEK == 23,'eid_week'] = 4
       x.loc[x.WEEK == 24,'eid_week'] = 5 
    elif year == 2019:
       x.loc[x.WEEK == 19,'eid_week'] = 1
       x.loc[x.WEEK == 20,'eid_week'] = 2
       x.loc[x.WEEK == 21,'eid_week'] = 3
       x.loc[x.WEEK == 22,'eid_week'] = 4
       x.loc[x.WEEK == 23,'eid_week'] = 5 
    return x



#For 真值計算數值
def rolling_mean_and_shift(x):
    x['bf5_mean'] = x['act_volume'].rolling(window=5).mean() 
    x['shift1_mean'] = x['bf5_mean'].shift(1)
    return x


#For 預測值計算數值
def rolling_mean_and_shift_predict(x):
    x['POD_Predict'] = x['POD_Predict'].apply(lambda x: np.round(x))
    x['bf5_mean_predict'] = x['POD_Predict'].rolling(window=5).mean()
    x['shift1_mean_predict'] = x['bf5_mean_predict'].shift(1)
    return x


#取得SKU開賣幾周
def get_act_week(x):
    x['act_week'] = np.arange(1,len(x)+1)
    return x
   
#取得lifecycle group
def get_life_cycle_band(Data):
    Data.loc[Data.act_week.isin(np.arange(1,10)),'lif_cycle_band'] = 1
    Data.loc[Data.act_week.isin(np.arange(10,20)),'lif_cycle_band'] = 2
    Data.loc[Data.act_week.isin(np.arange(20,30)),'lif_cycle_band'] = 3
    Data.loc[Data.act_week.isin(np.arange(30,40)),'lif_cycle_band'] = 4
    Data.loc[Data.act_week.isin(np.arange(40,50)),'lif_cycle_band'] = 5
    Data.loc[Data.act_week.isin(np.arange(50,60)),'lif_cycle_band'] = 6
    Data.loc[Data.act_week.isin(np.arange(60,9999)),'lif_cycle_band'] = 7
    return Data

#過濾數據outlier，計算偏權值
def filter_outlier(train_lifecycle_key):
    All_coffi_bias = []
    All_thisgroup_train = []
    for key2 ,key_value in tqdm(train_lifecycle_key):
        key_value = train_lifecycle_key.get_group(key2).copy()
        
        model = LinearRegression(fit_intercept=True)
        x = np.array(key_value['bf5_beid_mean'])
        model.fit(x[:, np.newaxis], np.array(key_value['Y_diff']))
        p30 = np.round(model.intercept_)
        key_value['bias'] = p30
        All_coffi_bias.append(p30)
        All_thisgroup_train.append(key_value)
    All_thisgroup_train_df = pd.concat(All_thisgroup_train)    
    ax = plt.boxplot(All_coffi_bias)    
    filter_value = ax['caps'][1].get_ydata()[0]
    plt.close()
    can_use_train_value1 = All_thisgroup_train_df[All_thisgroup_train_df['bias'] < filter_value]
    return can_use_train_value1


