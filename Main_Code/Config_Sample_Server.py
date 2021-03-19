# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:46:15 2019

@author: user
"""
import datetime
import sys

args = sys.argv

#Target time
if (len(args) > 2):
    Current_year = int(args[2].split('-')[0])
    Current_week = int(args[2].split('-')[1])
    print (("Target Week is :", args[2]))
else:
    Current_year = datetime.date.today().isocalendar()[0]
    Current_week = datetime.date.today().isocalendar()[1] 
#    Current_year = 2019
#    Current_week = 8
    print (("Target Week is :", str(Current_year) + '-' + str(Current_week).zfill(2)))


#設置
Country_List = ['GB','FR','TW','DE','RU','ID'] 
global safe_stock 
safe_stock = 20
state = 'current' #current or back
start_newweek = 261
filter_year = 2017
current_week = str(Current_year) + '_'+ str(Current_week).zfill(2)

#應映本機修改
User_path = '/All/'
User_NB_path = '/All/Project_NBStockManagement/'


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


#不用動
switch_path = './Switch/Final_result_30/'
POD_predict_path = './predict_result/'
origin_path = User_NB_path + 'predict/selling_result/XGB/WeeklyPrediction/'
output_path = User_NB_path + 'predict/selling_result/Switch_new_vol/'
POD_output_path = User_NB_path + '/predict/selling_result/POD_Model/'
Event_predict_path = POD_predict_path + 'predict_result_ID/'
Country_List_final = ['GB','FR','TW','DE','RU','ID','IE','IT','US']
origin_path = origin_predict_path