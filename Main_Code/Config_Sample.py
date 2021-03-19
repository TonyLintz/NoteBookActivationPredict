# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:46:15 2019

@author: user
"""
import datetime
import sys
import os

args = sys.argv

#Target time
if (len(args) > 1):
    Current_year = int(args[1].split('-')[0])
    Current_week = int(args[1].split('-')[1])
    print (("Target Week is :", args[1]))
else:
#    Current_year = datetime.date.today().isocalendar()[0]
#    Current_week = datetime.date.today().isocalendar()[1] 
    Current_year = 2020
    Current_week = 33
    print (("Target Week is :", str(Current_year) + '-' + str(Current_week).zfill(2)))

project_path = os.path.dirname(os.path.dirname(__file__))
pythonpaths = [
    os.path.join(project_path,'Main_Code/'),
]
for p in pythonpaths:
    sys.path.append(p)

#設置
#Country_List = ['GB','FR','TW','DE','RU','ID'] 
Country_List = ['ID'] 
global safe_stock 
safe_stock = 20
state = 'current' #current or back
start_newweek = 261
filter_year = 2017
current_week = str(Current_year) + '_'+ str(Current_week).zfill(2)

#應映本機修改
User_path = 'H:/All/'
User_NB_path = 'H:/All/Project_NBStockManagement/'


#ID 齋戒月時間 
EID_DICT = {2021:['2021-16','2021-17','2021-18','2021-19'],2020:['2020-18','2020-19','2020-20','2020-21'], 2019:['2019-20','2019-21','2019-22','2019-23'] , 2018:['2018-21','2018-22','2018-23','2018-24'], 2017:['2017-23','2017-24','2017-25','2017-26']}



#Read_Data_Path
Raw_Data_path = User_NB_path + 'predict/'
Share_Data_path = User_NB_path + 'ShareData/'
Countrylineup_path = User_path + 'Data_PnoCountryLineup/'
asus_week_path = User_path + 'Data_ProjectShareData/'
XGB_predict_path = User_NB_path + 'predict/selling_result/XGB/WeeklyPrediction/'
origin_predict_path = XGB_predict_path
Sellin_path = User_NB_path + 'NBStockResult/DataClean/integrate_raw/SellingPredict_ATAETA/'
log_path = os.path.join(project_path,'log/')

#不用動
switch_path = './Switch/Final_result_30/'
POD_predict_path = './predict_result/'
origin_path = User_NB_path + 'predict/selling_result/XGB/WeeklyPrediction/'
output_path = User_NB_path + 'predict/selling_result/Switch_new_vol/'
POD_output_path = User_NB_path + '/predict/selling_result/POD_Model/'
Event_predict_path = POD_predict_path + 'predict_result_ID/'
Country_List_final = ['GB','FR','TW','DE','RU','ID','IE','IT','US']
origin_path = origin_predict_path


Train_feature_columns = \
['l0name','MODEL_NAME', 'PART_NO','YEAR', 'WEEK','Y_label_yweek', 'Y_lable_by_week',\
 'Y_lable_by_week_log','before_1week_cum_act','before_1week_cum_pcs','gapweek', 'first_4_act_mean', \
 'until_act_mean','first_4_act_std', 'until_act_std','Fourth_act', 'Fiveth_act', 'change_slope_mean',\
   'First_slope', 'Second_slope', 'Third_slope', 'Fourth_slope','targetweek_actratio','targetweek_ratio_label',\
   'Model_first_4_act_mean', 'Model_until_act_mean','Model_Fourth_slope','before_1week_stock',\
   'W1_stock', 'W2_stock', 'W3_stock', 'W4_stock','W5_stock', 'W6_stock', 'W7_stock', 'W8_stock', 'W9_stock', \
   'W10_stock','W11_stock', 'W12_stock', 'W13_stock', 'W14_stock', 'W15_stock',\
   'W16_stock', 'W17_stock', 'W18_stock', 'W19_stock', 'W20_stock',\
   'Model_first_4_act_mean','act_week', 'normalize_act','EID_LABEL',\
   'first_4_MS_add', 'first_4_MS_Neg','until_MS_add','until_MS_Neg','sellin_freq','act_week_byyweek']



