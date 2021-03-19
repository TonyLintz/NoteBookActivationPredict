# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:34:56 2019

@author: Tony_Tien
"""

import pandas as pd
import os 
from tqdm import tqdm
import sys
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


#==============================================#
Folder_list = os.listdir(POD_predict_path)


#==============================================#

if __name__ == '__main__':
    
#===============選擇要處理的file，如current只取當周預測，如回朔則依照config設定===========================#
        for folder in Folder_list:
            
            #folder = 'predict_result_ID'
            Country = folder.split('_')[2]
            file_list = os.listdir(POD_predict_path + folder)
            
                
            if state == 'current':
                    
                    if (Current_week) > 53:
                        file_week = 1
                        file_year = Current_year + 1
                    else:
                        file_week = Current_week  
                        file_year = Current_year
                        
                    file_list = [str(file_year) + '_w' + str(file_week).zfill(2) + '.csv']
                 
            else:
                pass
                
                
            
            
#================Merge新舊模型預測===========================#
            Allfile = []
            for file in tqdm(file_list):
                   #file = '2019_w42.csv'
                   ata_predict = pd.read_csv(POD_predict_path + folder + '/' + file)[['l0name', 'model', 'pno', 'year', 'week', 'act_volume','POD_Predict']]
                   origin_predict = pd.read_csv(origin_path + file)[['l0name', 'model', 'pno', 'year', 'week', 'act_volume','predict']]
                   origin_predict = origin_predict[origin_predict['l0name'] == Country]
                   
                   
                   ss = pd.merge(ata_predict , origin_predict[['l0name', 'model', 'pno', 'year', 'week','predict']] ,on = ['l0name', 'model', 'pno', 'year', 'week'])
                   ss = ss[['l0name', 'model', 'pno', 'year', 'week', 'act_volume','predict','POD_Predict']]
                   if len(ss) != len(ata_predict):
                        print('=======================there are key no match===================================')  
                        
                        Allfile.append(file)
#                        break
                   ss.to_csv(POD_predict_path + folder + '/' + file,index=False)
      
#ss= ss[ss['predict'].isna() ==True]


#dd = pd.merge(ss , ata_predict , on = ['l0name', 'model', 'pno', 'year', 'week'],how='outer')
#ddd = dd[dd['predict'].isna()]
#dddd = ddd['pno'].unique().tolist()
#eol_sol_date = pd.read_csv(r'C:\Users\tony_tien\Desktop\W42\eol_sol_date_odd.csv')
#ssdsd = eol_sol_date[(eol_sol_date['PART_NO'].isin(dddd)) & (eol_sol_date['l0name'].isin(['ID']))]


#len(ata_predict.groupby(['l0name', 'model', 'pno']).size())
#len(origin_predict.groupby(['l0name', 'model', 'pno']).size())
  




#查POD model key  為何而少， 通常是因為當周已無激活量數據               
#oo = pd.merge(ata_predict , origin_predict[['l0name', 'model', 'pno', 'year', 'week','predict']] ,on = ['l0name', 'model', 'pno', 'year', 'week'],how= 'outer')
#ooo  = oo[oo['POD_Predict'].isna()]
#oooo = ooo.groupby(['l0name', 'model', 'pno']).filter(lambda x: len(x)==30)
#
#POD_model_loss_pno = oooo['pno'].unique().tolist()
#
#
##第一關確認  #75
#act_data[act_data['PART_NO'] == '90NR0112-M02610']
#
#All_no_predict_key = []
#for pno in POD_model_loss_pno:
#    #pno = '90NR0132-M05330'
#    data = NB_activation_byweektmp[NB_activation_byweektmp['PART_NO'] == pno]
#
#    what = data[(data['YEAR'] == 2019) & (data['WEEK'] == 42) ]
#    if len(what) == 0:
#        All_no_predict_key.append(pno)
#
#
#two_step_check = set(POD_model_loss_pno) - set(All_no_predict_key)
#
#
#
#act_data[act_data['PART_NO'] == '90NR0112-M02610']
