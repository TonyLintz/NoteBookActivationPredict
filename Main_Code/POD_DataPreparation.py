# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:54:40 2021

@author: Tony_Tien
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:41:33 2020

@author: Tony_Tien
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:26:20 2019

@author: Tony_Tien
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from utility.log import Log
import concurrent.futures

from Config import *
from function.Processing import getfirstday, fill_unitil_currentweek_for_notEOL, fillact,\
                                fill_tail, getact_insafe, get_ylabel_week, sellon_correct, EOL_cut
from function.Create_feature_func import back_to_school_feature, add_actweek_feature, future_sellin_feature, act_week30_forcurrentweek,\
                                Pno_value_statistics_feature, model_value_statistics_feature, week_weight_feature

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
log = Log(__name__).getlog()

def cut_eol_act_data(eol_sol_date, act_data):
    eol_sol_date_notyeteeol = eol_sol_date[eol_sol_date['EOL_YEAR'] != 9999].copy()
    act_data['week-date'] = act_data.apply(lambda x: getfirstday(x['YEAR'],x['WEEK']),axis=1)
    eol_sol_date_notyeteeol['eol_date'] = eol_sol_date_notyeteeol.apply(lambda x: getfirstday(x['EOL_YEAR'],x['EOL_WEEK']),axis=1)
    eol_sol_date_notyeteeol = eol_sol_date_notyeteeol.rename(columns = {'Model':'MODEL_NAME'})
    act_data_eol = pd.merge(act_data , eol_sol_date_notyeteeol[['l0name', 'PART_NO', 'eol_date']] , on = ['l0name', 'PART_NO'],how = 'left')
    act_data_have_eol = act_data_eol[~act_data_eol['eol_date'].isna()]
    act_data_no_eol = act_data_eol[act_data_eol['eol_date'].isna()]
    act_data_before_eol = act_data_have_eol[act_data_have_eol['week-date'] <= act_data_have_eol['eol_date']]
    act_data = pd.concat([act_data_no_eol , act_data_before_eol])
    act_data = act_data.sort_values(by = ['l0name' , 'MODEL_NAME' , 'PART_NO','YEAR','WEEK']).drop(['week-date', 'eol_date'],axis=1)
    return act_data


def newpno_filldata_process(act_data, asus_week_day, eol_sol_date):
    #針對不滿5周(4+當周)且還沒有EOL的key做補值，補到當周為止。(有些key如果在W27要預測，可能只有W23 W24 W25有資料，導致不滿五周會被以為還沒開賣)
    key_length = act_data.groupby(['l0name', 'MODEL_NAME', 'PART_NO']).apply(lambda x: len(x)).reset_index().rename(columns= {0:'sellon_length'} )
    small_5len_key = key_length[key_length['sellon_length'] <= 5]
    sellonend_data = pd.merge(small_5len_key , eol_sol_date , on = ['l0name', 'PART_NO'])
    not_EOL_small5 = sellonend_data[sellonend_data['EOL_YEAR'] == 9999]
    not_EOL_small5_group = not_EOL_small5.groupby(['l0name', 'MODEL_NAME', 'PART_NO'])
    
    Last_year = act_data['YEAR'].max()
    Last_week = act_data[act_data['YEAR'] == Last_year]['WEEK'].max()
    asus_week_day1 = asus_week_day[['year','week']].drop_duplicates().reset_index(drop=True)
    
    if len(not_EOL_small5) != 0:
        not_EOL_small5_afterfill = not_EOL_small5_group.apply(lambda x: fill_unitil_currentweek_for_notEOL(act_data , x , Last_year , Last_week, asus_week_day1)).reset_index().drop(['level_3'],axis=1)
        filter11 = act_data[~act_data['PART_NO'].isin(not_EOL_small5_afterfill['PART_NO'].unique().tolist())]
        act_data = pd.concat([filter11 , not_EOL_small5_afterfill]).sort_values(by = ['l0name' , 'MODEL_NAME' , 'PART_NO','YEAR','WEEK'])
    else:
        pass                
    return act_data, asus_week_day1


def filter_after_2017_sol(eol_sol_date, All_Data, act_data):
    is_vaild_sku = eol_sol_date[(eol_sol_date['SOL_YEAR']>= filter_year) & (eol_sol_date['l0name'] == Country) &(eol_sol_date['is_windows'] == 'Y')]
    All_Data = All_Data[All_Data['PART_NO'].isin(is_vaild_sku['PART_NO'])]
    act_data = act_data[act_data['PART_NO'].isin(is_vaild_sku['PART_NO'])]    
    pno_filter_2013_2014_sellin = All_Data[All_Data['YEAR'].isin([2013,2014])]['PART_NO'].tolist()
    All_Data = All_Data[~All_Data['PART_NO'].isin(pno_filter_2013_2014_sellin)]
    All_Data['YWEEK'] = All_Data.apply(lambda x : str(x['YEAR']) + '-'+ (str(x['WEEK']).zfill(2)),axis=1)
    return All_Data, act_data, is_vaild_sku


def fill_data_date(act_data, All_Data):
    act_data = act_data.groupby(['l0name','PART_NO'],as_index=False).apply(lambda x: fillact(x , asus_week_day))
    All_Data = All_Data.groupby(['l0name','PART_NO'],as_index=False).apply(lambda x: fillact(x , asus_week_day))
    act_data = act_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO','YEAR','WEEK']).reset_index(drop=True)
    All_Data = All_Data.sort_values(by = ['l0name', 'PART_NO','YEAR','WEEK']).reset_index(drop=True)
    return act_data, All_Data


def calcu_enrichment_info(All_Data):
    All_Data['cum_act'] = All_Data.groupby(['l0name', 'PART_NO'])['act_volume'].cumsum()
    All_Data['cum_pcs'] = All_Data.groupby(['l0name', 'PART_NO'])['pcs'].cumsum()
    All_Data['stock'] = All_Data['cum_pcs'] - All_Data['cum_act'] 
    
    All_Data['before_1week_cum_pcs'] = All_Data.groupby(['l0name', 'PART_NO'])['cum_pcs'].shift(1)
    All_Data['before_1week_cum_act'] = All_Data.groupby(['l0name', 'PART_NO'])['cum_act'].shift(1)
    All_Data['before_1week_stock'] = All_Data.groupby(['l0name', 'PART_NO'])['stock'].shift(1)
    All_value_df = All_Data.sort_values(by = ['l0name', 'PART_NO'])
    return All_value_df


def get_ylabel(Ready_Data, asus_week_day, asus_week_day1):
    #補尾巴
    asus_week_day = asus_week_day[['year','week','new_week']].drop_duplicates().reset_index(drop=False)
    Ready_Data_group = Ready_Data.groupby(['l0name','PART_NO'])
    
    All_output = []
    for key,value in tqdm(Ready_Data_group):
        output = fill_tail(value , asus_week_day)
        All_output.append(output)
    Ready_Data1 = pd.concat(All_output)
    Ready_Data_group = Ready_Data1.groupby(['l0name','PART_NO'])
    All_feature_df = Ready_Data_group.apply(lambda x: getact_insafe(x))
    All_feature_df = All_feature_df.reset_index(drop=True)
    
    All_gapweek = []
    for key,value in All_feature_df.groupby(['PART_NO','tag']):
        gapweek = np.arange(0,len(value))
        All_gapweek.extend(gapweek)
    All_feature_df['gapweek'] = All_gapweek   
    All_feature_df = All_feature_df.sort_values(by = ['l0name','PART_NO','YEAR','WEEK'])
    
    
    
    All_yweek_df = All_feature_df[['l0name','PART_NO','YEAR','WEEK']].drop_duplicates()
    All_Ylabel_week = get_ylabel_week(All_yweek_df,asus_week_day1)
    All_Ylabel_week['Y_label_yweek'] = All_Ylabel_week.apply(lambda x: str(x['year']) + '-' + str(x['week']).zfill(2),axis=1)
    All_feature_df = pd.concat([All_feature_df , All_Ylabel_week['Y_label_yweek']],axis=1)
    return All_feature_df, Ready_Data1


def process_rawdata_main(NB_activation_byweektmp_country, POD, asus_week_day, eol_sol_date):   
    
    log.info("Start: ready and process using table")    
    Model_Pno_mapping = NB_activation_byweektmp_country[['MODEL_NAME', 'PART_NO']].drop_duplicates() 
    pod_pno = POD[['pno']].drop_duplicates()
    act_pno = NB_activation_byweektmp_country[['PART_NO']].drop_duplicates()
    Can_use_key = set(pod_pno['pno']) & set(act_pno['PART_NO'])
    NB_activation_byweektmp_canuse = NB_activation_byweektmp_country[NB_activation_byweektmp_country['PART_NO'].isin(Can_use_key)]
    POD_canuse = POD[POD['pno'].isin(Can_use_key)].rename(columns = {'year':'YEAR' , 'week':'WEEK','pno':'PART_NO'})
    POD_data = POD_canuse.groupby(['l0name', 'PART_NO','YEAR', 'WEEK'],as_index=False)['pcs'].sum().sort_values(by = ['l0name', 'PART_NO' ,'YEAR', 'WEEK'])
    act_data = NB_activation_byweektmp_canuse.groupby(['l0name', 'MODEL_NAME', 'PART_NO','YEAR', 'WEEK'],as_index=False)['act_volume'].sum()
    act_data = act_data.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO' ,'YEAR', 'WEEK'])
    log.info("End: ready and process using table")
    
    #=======================cut eol data===============================================# 
    log.info("Start: cut eol data (first time)")
    act_data = cut_eol_act_data(eol_sol_date, act_data)
    act_data, asus_week_day1 = newpno_filldata_process(act_data, asus_week_day, eol_sol_date)
    log.info("End:  cut eol data (first time)")
    
    #=======================merge actdata and poddata = All_Data=======================# 
    log.info("Start: merge actdata and poddata")
    All_Data = pd.merge(act_data , POD_data , on = ['l0name', 'PART_NO','YEAR', 'WEEK'] , how ='outer')
    All_Data['pcs'] = All_Data['pcs'].fillna(0)
    All_Data['act_volume'] = All_Data['act_volume'].fillna(0)
    All_Data = All_Data.sort_values(by = ['l0name', 'PART_NO','YEAR', 'WEEK'])[['l0name','PART_NO','YEAR','WEEK','act_volume','pcs']]
    log.info("End: merge actdata and poddata")
    
    #=======================篩2017開始且屬於windows的SKU========================================#
    log.info("Start: Screening data after 2017 and is windows")
    All_Data, act_data, is_vaild_sku = filter_after_2017_sol(eol_sol_date, All_Data, act_data)    
    log.info("End: Screening data after 2017 and is windows")
    #=======================補齊資料間空缺時間==================================================#
    log.info("Start: fill date")
    act_data, All_Data = fill_data_date(act_data, All_Data)
    log.info("End: fill date")
    #=======================增加累加量及歷史資訊================================================#
    log.info("Start: get cumsum and history info")
    All_value_df = calcu_enrichment_info(All_Data)    
    log.info("End: get cumsum and history info")
    #========================切開賣後5周=======================================================#
    log.info("Start: Screen out the data five weeks after the start of the sale")
    All_value_df2 = sellon_correct(All_value_df,Can_use_key)
    log.info("End: Screen out the data five weeks after the start of the sale")
    #========================再切一次EOL=======================================================#
    log.info("Start: cut eol data (second time)")
    All_fillact_value_df = All_value_df2.copy()
    Ready_Data = EOL_cut(All_fillact_value_df , act_data)
    log.info("End: cut eol data (second time)")
    #========================================製作Y_Label=======================================#
    log.info("Start: create ylabel")
    All_feature_df, Ready_Data1 = get_ylabel(Ready_Data, asus_week_day, asus_week_day1)   
    log.info("End: create ylabel")
    return All_feature_df, All_fillact_value_df, Ready_Data1, act_data, is_vaild_sku, Model_Pno_mapping
    

def creat_feature_main(All_feature_df, All_fillact_value_df, Ready_Data1, act_data, is_vaild_sku, Model_Pno_mapping):   
        
    #=======back to school特徵==========#
    log.info("Start: create back to school feature")
    All_add_future_sellin = back_to_school_feature(All_fillact_value_df, All_feature_df)
    log.info("End: create back to school feature")
    #========針對當周，未來的20周的actweek================#
    log.info("Start: create actweek(for future)")
    All_add_future_sellin = add_actweek_feature(All_add_future_sellin)
    log.info("End: create actweek(for future)")
    #========未來庫存特徵================#
    log.info("Start: create future sellin feature")
    Feature_set = future_sellin_feature(Ready_Data1, All_fillact_value_df)
    log.info("End: create future sellin feature")
    #============當周的actweek是多少=========================#
    log.info("Start: create actweek feature(for current)")
    All_new_value_df = act_week30_forcurrentweek(Feature_set)    
    log.info("End:  create actweek feature(for current)")
    #========== Pno數值特徵 ==============================================#
    log.info("Start: create pno statistics feature")
    Feature_set = Pno_value_statistics_feature(act_data, All_new_value_df)
    log.info("End: create pno statistics feature")
    #========== Model數值特徵=============================================#
    log.info("Start: create model statistics feature")
    New_Feature_df2 = model_value_statistics_feature(act_data, Feature_set, Model_Pno_mapping)
    log.info("Start: create model statistics feature")

    #Model_first_4_act_slope、Model_first_4_act將欄位展開成多個特徵欄位
    log.info("Start: Expand field")
    Model_first_4_act_slope_df = pd.DataFrame(New_Feature_df2['Model_first_4_act_slope'].values.tolist() , columns = ['Model_First_slope','Model_Second_slope','Model_Third_slope','Model_Fourth_slope'])
    Model_first_4_act_df = pd.DataFrame(New_Feature_df2['Model_first_4_act'].values.tolist() , columns = ['Model_First_act','Model_Second_act','Model_Third_act','Model_Fourth_act','Model_Fiveth_act'])
    New_Feature_df2 = New_Feature_df2.reset_index(drop=True)
    New_Feature_df3 = pd.concat([New_Feature_df2 , Model_first_4_act_slope_df , Model_first_4_act_df ],axis=1)
    New_Feature_df3 = New_Feature_df3.drop(['Model_first_4_act_slope' , 'Model_first_4_act'],axis=1)
    log.info("End: Expand field")

    #==============Merge 周格式特徵及SKU格式特徵=============================================================================================#   
    log.info("Start: Expand field")
    Feature_set_new = pd.merge(All_add_future_sellin[['l0name', 'PART_NO', 'YEAR', 'WEEK','gapweek','Y_label_yweek','Y_lable_by_week','before_1week_cum_act','before_1week_cum_pcs','targetweek_actratio', 'targetweek_ratio_label','act_week_byyweek']] , New_Feature_df3 , on = ['l0name','PART_NO','YEAR', 'WEEK'])
    log.info("End: Expand field")
    #===================加入周權重 & 齋戒月及開齋節Label型特徵(齋戒月:1、開齋節:2、其他:0)==========================================================================================================#
    log.info("Start: create weekly weight and EID lalel")
    All_YEAR_WEEK_FEATURE_DF = week_weight_feature(Feature_set_new, act_data, is_vaild_sku, EID_DICT)
    log.info("End: create weekly weight and EID lalel")

    Feature_set = All_YEAR_WEEK_FEATURE_DF.sort_values(by = ['l0name', 'PART_NO', 'YEAR', 'WEEK','gapweek','Y_label_yweek']).drop(['Y_label_year'],axis=1)

    return Feature_set



if __name__ == '__main__':
    log.info("Start: ---------------POD_DataPreparation.py--------------------")

    log.info("Start: Read raw data")
    NB_activation_byweektmp = pd.read_csv(Raw_Data_path  +'NB_activation_byweektmp.csv')[['l0name', 'MODEL_NAME', 'PART_NO', 'CUSTOMER','YEAR','WEEK', 'ActWeek','act_volume']]
    pod = pd.read_csv(Sellin_path + '{}'.format(current_week) + '/selling_all.csv')
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','date','new_week']]
    eol_sol_date = pd.read_csv(Share_Data_path + 'eol_sol_date.csv')
    log.info("End: Read raw data")
    
    for Country in Country_List:
        log.info("Start: start {} part".format(Country))
        NB_activation_byweektmp_country = NB_activation_byweektmp[NB_activation_byweektmp['l0name'] == Country]
        POD = pod[pod['l0name'] == Country]
        All_feature_df, All_fillact_value_df, Ready_Data1, act_data, is_vaild_sku, Model_Pno_mapping = process_rawdata_main(NB_activation_byweektmp_country, POD, asus_week_day, eol_sol_date)
        Feature_set = creat_feature_main(All_feature_df, All_fillact_value_df, Ready_Data1, act_data, is_vaild_sku, Model_Pno_mapping)
        Feature_set.to_csv(project_path + '/Data/Feature_set_'+Country+'.csv',index=False,encoding='utf-8')
        log.info("End: End {} part".format(Country))
    
    log.info("End: ---------------POD_DataPreparation.py--------------------")

