# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:33:42 2021

@author: Tony_Tien
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import heapq

from Config import *
from function.Calculate import add_act_week, add_future_stock, get_sku_sellin_length_table, get_mean_feature,\
                                        get_slope_feature, feature_set_clean, get_model_mean_feature,get_model_slope_feature


def back_to_school_feature(All_fillact_value_df, All_feature_df):

    #==========================================back to school特徵============================================#
    All_fillact_value_df = All_fillact_value_df.rename(columns = {'YWEEK':'Y_label_yweek'})
#    All_fillact_value_df = All_fillact_value_df[All_fillact_value_df['Y_label_yweek'] <= (str(Current_year) + '-' + str(Current_week).zfill(2))]
    All_add_future_sellin = pd.merge(All_feature_df[['l0name', 'PART_NO', 'YEAR', 'WEEK','gapweek','Y_label_yweek' ,'Y_lable_by_week']] , All_fillact_value_df[['l0name', 'PART_NO', 'Y_label_yweek','before_1week_cum_act', 'before_1week_cum_pcs']]  , on = ['l0name','PART_NO','Y_label_yweek'],how = 'left')
    
    All_add_future_sellin.loc[All_add_future_sellin.Y_label_yweek > (str(Current_year) + '-' + str(Current_week).zfill(2)) , 'before_1week_cum_act'] = np.nan
    
    All_add_future_sellin['targetweek_actratio'] = All_add_future_sellin['before_1week_cum_act'] / All_add_future_sellin['before_1week_cum_pcs']
    All_add_future_sellin.loc[All_add_future_sellin.targetweek_actratio.isna(),'targetweek_actratio'] = 1
    

    current_condition = All_add_future_sellin.groupby(['l0name', 'PART_NO','YEAR','WEEK'],as_index=False).first()
    current_condition.loc[current_condition.targetweek_actratio > 0.6,'current_ratio_label'] = 1
    current_condition.loc[current_condition.targetweek_actratio <= 0.6,'current_ratio_label'] = 0

    All_add_future_sellin = pd.merge(All_add_future_sellin , current_condition[['l0name', 'PART_NO', 'YEAR', 'WEEK','current_ratio_label']] , on = ['l0name', 'PART_NO','YEAR','WEEK']) 
    

    All_add_future_sellin.loc[(All_add_future_sellin.current_ratio_label == 1) & (All_add_future_sellin.targetweek_actratio >= 0.84)  , 'targetweek_ratio_label'] = 1
    All_add_future_sellin.loc[(All_add_future_sellin.current_ratio_label == 1) & (All_add_future_sellin.targetweek_actratio < 0.84)  , 'targetweek_ratio_label'] = 0
    All_add_future_sellin.loc[(All_add_future_sellin.current_ratio_label == 0) & (All_add_future_sellin.targetweek_actratio >= 0.75)  , 'targetweek_ratio_label'] = 1
    All_add_future_sellin.loc[(All_add_future_sellin.current_ratio_label == 0) & (All_add_future_sellin.targetweek_actratio < 0.75)  , 'targetweek_ratio_label'] = 0
    All_add_future_sellin = All_add_future_sellin.drop(['current_ratio_label'],axis=1)
    return All_add_future_sellin


def add_actweek_feature(data):
    SKUweekDF = data[['l0name','PART_NO', 'YEAR', 'WEEK']].drop_duplicates()
    actweek_df = SKUweekDF.groupby(['l0name','PART_NO']).apply(lambda x: add_act_week(x))
    actweek_df['Y_label_yweek'] = actweek_df.apply(lambda x: str(x['YEAR']) + '-' + str(x['WEEK']).zfill(2),axis=1)     
    data.loc[SKUweekDF.index,'act_week'] = actweek_df['act_week'] 
    data['act_week'] = data['act_week'].fillna(1)
    data['real_act_week'] = data.groupby(['l0name','PART_NO', 'YEAR', 'WEEK'])['act_week'].cumsum()
    data = data.drop(['act_week'],axis=1).rename(columns = {'real_act_week':'act_week_byyweek'})
    return data


def future_sellin_feature(Ready_Data1, All_fillact_value_df):
    future_sellin = pd.merge(Ready_Data1 , All_fillact_value_df[['l0name', 'PART_NO', 'YEAR', 'WEEK','pcs']] , on = ['l0name','PART_NO','YEAR','WEEK'],how = 'left')
    add_future_stock_list = Ready_Data1.groupby(['l0name','PART_NO']).apply(lambda x: add_future_stock(x , future_sellin)).reset_index().rename(columns = {0:'future_stock'})    
    All_SKU_sellinfreq_df = Ready_Data1.groupby(['l0name','PART_NO']).apply(lambda x: get_sku_sellin_length_table(x , future_sellin)).reset_index()

    for pno in tqdm(add_future_stock_list.PART_NO):
        Ready_Data1.loc[Ready_Data1['PART_NO'] == pno,'future_stock'] = add_future_stock_list[add_future_stock_list['PART_NO'] == pno]['future_stock'].iloc[0]

    Ready_Data1 = Ready_Data1.fillna('')
    Feature_set = Ready_Data1[Ready_Data1['YWEEK'] != '']
    future_stock_feature = pd.DataFrame(Feature_set['future_stock'].values.tolist())
    Feature_set = Feature_set.reset_index()
    
    col_list = []
    for i in range(1,safe_stock+1):
            col_list.append('W'+str(i)+'_stock')
            
    future_stock_feature.columns = col_list
    Feature_set = pd.concat([Feature_set , future_stock_feature],axis=1).drop(['index','future_stock'],axis=1)     
    Feature_set = pd.merge(Feature_set , All_SKU_sellinfreq_df , on =['l0name', 'PART_NO', 'YEAR', 'WEEK'])
    return Feature_set


def act_week30_forcurrentweek(Feature_set):
    Feature_set = Feature_set.groupby(['l0name','PART_NO']).apply(lambda x: add_act_week(x))
    Feature_set = Feature_set.sort_values(by = ['l0name', 'PART_NO', 'YEAR', 'WEEK']) 
    return Feature_set


def Pno_value_statistics_feature(act_data, All_new_value_df):
    #製作平均特徵
    act_data_group = act_data.groupby(['l0name','MODEL_NAME','PART_NO'])
    Act_mean_data_df = act_data_group.apply(lambda x: get_mean_feature(x))
    Act_mean_data_df = Act_mean_data_df.sort_values(by = ['l0name','PART_NO','YEAR','WEEK'])[['l0name','MODEL_NAME','PART_NO','YEAR','WEEK','first_4_act','first_4_act_mean','until_act_mean','first_4_act_std','until_act_std']]
    All_new_value_df1 = pd.merge(Act_mean_data_df , All_new_value_df , on = ['l0name','PART_NO','YEAR','WEEK'],how='inner')
    
    #製作變化率特徵
    Act_slope_data_df = act_data_group.apply(lambda x: get_slope_feature(x))
    Act_slope_data_df = Act_slope_data_df.sort_values(by = ['l0name','PART_NO','YEAR','WEEK'])[['l0name','PART_NO','YEAR','WEEK','change_slope_mean','first_4_act_slope']]
    All_new_value_df2 = pd.merge(Act_slope_data_df[['l0name','PART_NO','YEAR','WEEK','first_4_act_slope','change_slope_mean']] , All_new_value_df1  ,on = ['l0name','PART_NO','YEAR','WEEK'],how='inner')
    
    #特徵集整理
    Feature_set = feature_set_clean(All_new_value_df2)
    
    return Feature_set


def model_value_statistics_feature(act_data, Feature_set, Model_Pno_mapping):
    act_model = act_data.groupby(['l0name','MODEL_NAME','YEAR','WEEK'],as_index=False)['act_volume'].sum()
    act_data_model_group = act_model.groupby(['l0name','MODEL_NAME'])
    Model_Act_mean_data_df = act_data_model_group.apply(lambda x: get_model_mean_feature(x))
    Model_Act_mean_data_df = Model_Act_mean_data_df.sort_values(by = ['l0name','MODEL_NAME','YEAR','WEEK'])[['l0name','MODEL_NAME','YEAR','WEEK','first_4_act','first_4_act_mean','until_act_mean','first_4_act_std','until_act_std']].rename(columns={'first_4_act':'Model_first_4_act' , 'first_4_act_mean':'Model_first_4_act_mean' , 'until_act_mean':'Model_until_act_mean' , 'first_4_act_std':'Model_first_4_act_std' , 'until_act_std':'Model_until_act_std'})
    Model_Act_slope_data_df = act_data_model_group.apply(lambda x: get_model_slope_feature(x))
    Model_Act_slope_data_df = Model_Act_slope_data_df.sort_values(by = ['l0name','MODEL_NAME','YEAR','WEEK'])[['l0name','MODEL_NAME','YEAR','WEEK','change_slope_mean','first_4_act_slope']].rename(columns={'change_slope_mean':'Model_change_slope_mean' , 'first_4_act_slope': 'Model_first_4_act_slope'})
    Feature_set = pd.merge(Model_Pno_mapping , Feature_set , on = ['PART_NO'])
    Feature_set['YEAR'] = Feature_set['YEAR'].astype(int)
    Feature_set['WEEK'] = Feature_set['WEEK'].astype(int)
    #============Merge Model Mean and change rate================#
    New_Feature_df1 = pd.merge(Feature_set , Model_Act_mean_data_df , on = ['l0name', 'MODEL_NAME', 'YEAR', 'WEEK'])
    New_Feature_df2 = pd.merge(New_Feature_df1 , Model_Act_slope_data_df , on = ['l0name', 'MODEL_NAME', 'YEAR', 'WEEK'])
    New_Feature_df2 = New_Feature_df2.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK'])
    return New_Feature_df2



def week_weight_feature(Feature_set_new, act_data, is_vaild_sku, EID_DICT):

    Feature_set_new['Y_label_week'] = Feature_set_new.apply(lambda x: int(x['Y_label_yweek'].split('-')[1]),axis=1)
    Feature_set_new['Y_label_year'] = Feature_set_new.apply(lambda x: int(x['Y_label_yweek'].split('-')[0]),axis=1)
#    Feature_set = pd.merge(Feature_set_new , year_week_act[['Y_label_week','normalize_act']] , on =['Y_label_week'])
    
    
    All_YEAR_WEEK_FEATURE = []
    for YYY in Feature_set_new['Y_label_year'].unique():
        
        if YYY < Current_year | YYY > Current_year:
            act_data_valid = act_data[(act_data['PART_NO'].isin(is_vaild_sku['PART_NO'])) & (act_data['YEAR'] == YYY)]
            year_week_act = act_data_valid.groupby(['WEEK'],as_index=False)['act_volume'].sum()
            year_week_act['normalize_act'] = year_week_act['act_volume'] / year_week_act['act_volume'].sum()
            year_week_act = year_week_act.rename(columns = {'WEEK':'Y_label_week'})
        
            current_year_data = Feature_set_new[Feature_set_new['Y_label_year'] == YYY]
            current_year_data1 = pd.merge(current_year_data , year_week_act[['Y_label_week','normalize_act']] , on =['Y_label_week'])
            
            current_year_data1.loc[current_year_data1.Y_label_week.isin(EID_DICT[YYY][0:3]) , 'EID_LABEL'] = 1
            current_year_data1.loc[current_year_data1.Y_label_week.isin([EID_DICT[YYY][-1]]) , 'EID_LABEL'] = 2
            current_year_data1.loc[~current_year_data1.Y_label_week.isin(EID_DICT[YYY]) , 'EID_LABEL'] = 0
        else:
            act_data_valid = act_data[(act_data['PART_NO'].isin(is_vaild_sku['PART_NO'])) & (act_data['YEAR'] == Current_year-1)]
            year_week_act = act_data_valid.groupby(['WEEK'],as_index=False)['act_volume'].sum()
            year_week_act['normalize_act'] = year_week_act['act_volume'] / year_week_act['act_volume'].sum()
            year_week_act = year_week_act.rename(columns = {'WEEK':'Y_label_week'})
            Eid_weight = heapq.nsmallest(4, year_week_act['normalize_act'].tolist())
        
            

            YEAR_EID_WEEK = EID_DICT[YYY]  
            EID_WEEK_LIST = list(map(lambda x: int(x.split('-')[1]),YEAR_EID_WEEK))
            
            first_back_to_school_week = 30
            Next_one_EID_week = max(EID_WEEK_LIST) + 1
            
            smallest_weight_next_one_index = year_week_act['normalize_act'].idxmin() + 1
            smallest_weight_next_one_weight = year_week_act.loc[smallest_weight_next_one_index,'normalize_act']
            Back_to_school_index = year_week_act[year_week_act['Y_label_week'] == first_back_to_school_week].index[0]
            Back_to_school_weight = year_week_act.loc[Back_to_school_index,'normalize_act']
            
            gap = first_back_to_school_week - Next_one_EID_week + 1
            
            EID_TO_BTS_WEIGHT = np.linspace(smallest_weight_next_one_weight,Back_to_school_weight,gap)
            year_week_act.loc[year_week_act.Y_label_week.isin(np.arange(Next_one_EID_week , first_back_to_school_week+1)),'normalize_act'] = EID_TO_BTS_WEIGHT
            
            
            year_week_act.loc[year_week_act.Y_label_week == EID_WEEK_LIST[3],'normalize_act'] = Eid_weight[0]
            year_week_act.loc[year_week_act.Y_label_week == EID_WEEK_LIST[2],'normalize_act'] = Eid_weight[2]
            year_week_act.loc[year_week_act.Y_label_week == EID_WEEK_LIST[1],'normalize_act'] = Eid_weight[1]
            year_week_act.loc[year_week_act.Y_label_week == EID_WEEK_LIST[0],'normalize_act'] = Eid_weight[3]
         
            empty = pd.Series()
            empty['Y_label_week'] = int(53)
            empty['normalize_act'] = year_week_act['normalize_act'].iloc[0]
            empty = pd.DataFrame(empty).transpose()
            year_week_act = pd.concat([year_week_act , empty])
            
            
            current_year_data = Feature_set_new[Feature_set_new['Y_label_year'] == YYY]
            current_year_data1 = pd.merge(current_year_data , year_week_act[['Y_label_week','normalize_act']] , on =['Y_label_week'])
            
            current_year_data1.loc[current_year_data1.Y_label_week.isin(EID_DICT[YYY][0:3]) , 'EID_LABEL'] = 1
            current_year_data1.loc[current_year_data1.Y_label_week.isin([EID_DICT[YYY][-1]]) , 'EID_LABEL'] = 2
            current_year_data1.loc[~current_year_data1.Y_label_week.isin(EID_DICT[YYY]) , 'EID_LABEL'] = 0

        All_YEAR_WEEK_FEATURE.append(current_year_data1)
    All_YEAR_WEEK_FEATURE_DF = pd.concat(All_YEAR_WEEK_FEATURE)
    return All_YEAR_WEEK_FEATURE_DF