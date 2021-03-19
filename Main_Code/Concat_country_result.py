# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:32:47 2019

@author: Tony_Tien
"""

import pandas as pd
import numpy as np
import os 
import sys
import shutil
from tqdm import tqdm

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

if __name__ == '__main__':

    file_country_name = os.listdir(switch_path)
    predict_path = switch_path + file_country_name[0] + '/'
    #switch_predict_file_list = os.listdir(switch_path + file_country_name[0])
    
    if state == 'current':
            if Current_week > 53:
                file_list = [str(Current_year)+'_w'+str(1).zfill(2)+'.csv']
            else:
                file_list = [str(Current_year)+'_w'+str(Current_week).zfill(2)+'.csv']
            
    elif state == 'back':
            predict_file = os.listdir(predict_path)
            file_list = list(filter(lambda x:  x.endswith(".csv") ,predict_file))
    
    
    
    
    for file in file_list:
            #file = '2019_w32.csv'
            print(file)
            
            All_country_result = pd.DataFrame()
            for country in Country_List_final:
                #country = 'GB'
               
                try:
                    origin_result = pd.read_csv(origin_path + file)
                    switch_result  = pd.read_csv(switch_path + 'Final_result_'+country+'/'+file) 
                    All_country_result = pd.concat([All_country_result , switch_result],axis=0)
                
                except:
                   
                    
                    origin_result_country = origin_result[origin_result['l0name'] == country]
                    All_country_result = pd.concat([All_country_result , origin_result_country],axis=0)

            if len(All_country_result) != len(origin_result):
                print('========no match========')
                break
            All_country_result['predict'] = All_country_result['predict'].apply(lambda x: np.round(x))
            All_country_result.to_csv(output_path + file,index=False,encoding='utf-8')
            
            


    origin_file_list = os.listdir(output_path)
    
    for file in tqdm(origin_file_list):
        origin_result = pd.read_csv(origin_path + file)
        switch_result = pd.read_csv(output_path + file)
        
    
        if len(origin_result) != len(switch_result):
            print('========no match========')
            break
    
    
    
    
    
    
#    Origin_predict_list = os.listdir(origin_path)
#    Origin_predict_list1 = list(filter(lambda x:  x.endswith(".csv") ,Origin_predict_list))
#     
#    POD_predict_list = os.listdir(output_path)
#    POD_predict_list1 = list(filter(lambda x:  x.endswith(".csv") ,POD_predict_list))
#    
#    
#    for file in tqdm(POD_predict_list1):
#         
#         origin_predict = pd.read_csv(origin_path + file)
#         origin_predict = origin_predict.sort_values(by = ['l0name', 'model', 'pno', 'year', 'week']).reset_index(drop=True)
#         switch_predict = pd.read_csv(output_path + file)
#         switch_predict = switch_predict.sort_values(by = ['l0name', 'model', 'pno', 'year', 'week']).reset_index(drop=True)
#         
#  
#         if len(origin_predict) != len(switch_predict):
#                  break
#         
#         switch_predict['act_volume'] = origin_predict['act_volume']
#         
#         if any(origin_predict['act_volume'] != switch_predict['act_volume']) :
#                 break
#         
#         switch_predict.to_csv(output_path + file,index=False,encoding='utf-8')
#    
    