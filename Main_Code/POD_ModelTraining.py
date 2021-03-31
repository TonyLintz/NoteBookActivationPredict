# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:42:22 2020

@author: Tony_Tien
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import sys
import re
import shutil
import math
#================ML package==========================#

from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression , chi2 ,GenericUnivariateSelect
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import  ensemble, preprocessing, metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest ,SelectPercentile ,mutual_info_regression,f_regression
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model, datasets, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer
from sklearn import cluster, datasets
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from Config import *
from utility.log import Log
log = Log(__name__).getlog()    
#=================================================#

def negative_address(Predict_result):
    Predict_result = np.array(Predict_result)
    All_array = []
    for i in Predict_result:
        i[i <  0] = 0
        All_array.append(i)
    All_array = np.array(All_array)
    return All_array


def expY(series_y_list):
     series_y_list = list(map(lambda x: np.expm1(x),series_y_list))
     return series_y_list


def logY(series_y_list):
     series_y_list = list(map(lambda x: np.log1p(x),series_y_list))
     return series_y_list
 
    
def custom_asymmetric_objective(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual>0, -2*1*residual, -2*residual)
    hess = np.where(residual>0, 2*1, 2)
    return grad, hess


def custom_asymmetric_eval(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual > 0, (residual**2)*1, residual**2) 
    return "custom_asymmetric_eval", np.mean(loss), False


def predict_thisweek(Train_data_week ,Test_data_week, gapweek):
   
    Train_data = Train_data_week.copy()
    Test_data = Test_data_week.copy()
    
    train_test_data = pd.concat([Train_data,Test_data],axis=0)
    pretraining_data_s = preprocessing.scale(train_test_data.iloc[:,11::].values)
    

    X = pretraining_data_s[:len(Train_data)]
    y = np.array(Train_data['Y_lable_by_week_log'].values.tolist())

    X_train_1, X_valid, y_train_1, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    X_train = pretraining_data_s[:len(Train_data)]
    X_test = pretraining_data_s[len(Train_data)::]
    y_train = Train_data['Y_lable_by_week_log'].values.tolist()
    y_test = Test_data['Y_lable_by_week_log'].values.tolist()
    
    if gapweek > -1:
        gbm6 = lgb.LGBMRegressor(random_state=465,alpha = 0.7,n_estimators=180,boosting='gbdt',learning_rate=0.09, min_data_in_leaf= 25,num_leaves=25,max_depth =5 ,bagging_fraction = 0.2,feature_fraction = 0.5)#bagging_fraction = 0.2  feature_fraction = 0.5
        gbm6.set_params(**{'objective': 'quantile'}, metrics = ["quantile",'huber'])        
        X_train = pretraining_data_s[:len(Train_data)]
        X_test = pretraining_data_s[len(Train_data)::]
        y_train = Train_data['Y_lable_by_week_log'].values.tolist()
        y_test = Test_data['Y_lable_by_week_log'].values.tolist()
        gbm6.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['quantile','huber'],verbose = False)

    else:
        gbm6 = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.097,n_estimators=130 , random_state = 465,min_child_samples=30, min_child_weight=0.01)
        gbm6.fit(X_train , y_train)

    Predict_result = gbm6.predict(X_test)
    Predict_result = list(map(lambda x: np.expm1(x),Predict_result))
    return Predict_result,Test_data,y_test


def predict_file_format(All_key_predict_result_df):
    All_key_predict_result_df = All_key_predict_result_df.drop(['YEAR','WEEK','gapweek'],axis=1)
    All_key_predict_result_df.loc[All_key_predict_result_df.predict_result < 0,'predict_result'] = 0
    All_key_predict_result_df['year'] = All_key_predict_result_df.apply(lambda x: str(x['Y_label_yweek'].split('-')[0]) , axis=1)
    All_key_predict_result_df['week'] = All_key_predict_result_df.apply(lambda x: str(x['Y_label_yweek'].split('-')[1]) , axis=1)
    All_key_predict_result_df = All_key_predict_result_df.rename(columns = {'MODEL_NAME':'model','PART_NO':'pno','predict_result':'POD_Predict' ,'Y_lable_by_week':'act_volume'}).drop(['Y_label_yweek'],axis=1)
    All_key_predict_result_df = All_key_predict_result_df[['l0name','model', 'pno','year', 'week' , 'act_volume' , 'POD_Predict']]
    return All_key_predict_result_df


def rolling_feature_and_predict_oneweek(Feature_set_select, Test_data, Target_Year, Target_Week):

        Predict_result = []
        weekly_predict_result = [] 
        for next_x_predict in tqdm(np.arange(safe_stock)):
            
            #切可使用的Train 
            if (Target_Week-next_x_predict - 1) <= 0:
                Train_until_week = Target_Week - next_x_predict - 1 + 53
                Train_until_year = Target_Year - 1
            else:
                Train_until_week = Target_Week - next_x_predict - 1 
                Train_until_year = Target_Year 
                
            Train_data = Feature_set_select[((Feature_set_select['YEAR'] == Train_until_year) & (Feature_set_select['WEEK'] <= Train_until_week)) | ((Feature_set_select['YEAR'] < Train_until_year))]
            Train_data_week = Train_data[Train_data['gapweek'] == next_x_predict]
            Train_data_week = Train_data_week.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','Y_label_yweek'])
            Train_data_week = Train_data_week.replace([np.inf, -np.inf],1)
            Train_data_week = Train_data_week.dropna()
            Test_data_week = Test_data[Test_data['gapweek'] == next_x_predict]
            
            
            if next_x_predict == 0:
                Test_data_week = Test_data_week.sort_values(by= ['l0name', 'MODEL_NAME', 'PART_NO'])

            else:
#                        Test_data_week = Test_data_week.sort_values(by= ['l0name', 'MODEL_NAME', 'PART_NO']).drop(['before_1week_cum_act','act_shift1','act_shift2'],axis=1)
                Test_data_week = Test_data_week.sort_values(by= ['l0name', 'MODEL_NAME', 'PART_NO']).drop(['before_1week_cum_act'],axis=1)
                Test_data_week['before_1week_cum_act'] = ssssa['Nextweek_before_1week_cum_act'].values.tolist()
                Test_data_week['targetweek_actratio'] = Test_data_week['before_1week_cum_act'] / Test_data_week['before_1week_cum_pcs']
                Test_data_week.loc[(Test_data_week.targetweek_actratio >= 0.84)  , 'targetweek_ratio_label'] = 1
                Test_data_week.loc[(Test_data_week.targetweek_actratio < 0.84)  , 'targetweek_ratio_label'] = 0
#                        Test_data_week['act_shift1'] = ssssa['predict_result'].values.tolist()
#                        Test_data_week['act_shift2'] = ssssa['act_shift1'].values.tolist()
#                        
#                        Test_data_week = Test_data_week.drop(['targetweek_ratio_label'],axis=1)
            Test_data_week = Test_data_week.replace([np.inf, -np.inf],1)            
            Predict_result , Test_data_week , y_test = predict_thisweek(Train_data_week[Train_feature_columns] ,Test_data_week[Train_feature_columns] , next_x_predict)
            Test_data_week['predict_result'] = Predict_result
            ssssa = Test_data_week[['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','gapweek','Y_label_yweek','Y_lable_by_week','before_1week_cum_act','before_1week_cum_pcs','predict_result']].copy()
            ssssa['predict_result'] = np.floor(ssssa['predict_result']) 
            ssssa = ssssa.sort_values(by= ['l0name', 'MODEL_NAME', 'PART_NO'])
            ssssa['Nextweek_before_1week_cum_act'] = ssssa['before_1week_cum_act'] + ssssa['predict_result']
            weekly_predict_result.append(ssssa[['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','gapweek','Y_label_yweek','Y_lable_by_week','before_1week_cum_act','before_1week_cum_pcs','Nextweek_before_1week_cum_act','predict_result']])
        return weekly_predict_result




if __name__ == '__main__':
    log.info("Start: ---------------POD_ModelTraining.py--------------------")

    
    Data_file = os.listdir(os.path.join(project_path,'Data/'))
    pattern = re.compile(r'Feature_set_[A-Z]{2}')
    Feature_set_name = list(filter(lambda x: bool(pattern.match(x)) == True,Data_file))
    eol_sol_date = pd.read_csv(Share_Data_path + 'eol_sol_date.csv')
    asus_week_day = pd.read_csv(asus_week_path + 'asus_week_day.csv')[['year','week','new_week']].drop_duplicates().reset_index(drop=True)
    
    after_2017_sol_key = eol_sol_date[(eol_sol_date.SOL_YEAR >= 2017) & (eol_sol_date.is_windows == 'Y')]
    is_windows = eol_sol_date[eol_sol_date['is_windows'] == 'Y']
    
    os.path.join(project_path,'predict_result/')
    if (not os.path.exists(os.path.join(project_path,'predict_result/'))): 
            os.mkdir(os.path.join(project_path,'predict_result/'))
    
    for featureset in Feature_set_name:
        #featureset = 'Feature_set_ID.csv'
        Feature_set = pd.read_csv(os.path.join(project_path,'Data/') + featureset)
        Country = Feature_set['l0name'].iloc[0]
        log.info("Start: {} ModelTraining".format(Country))
        
        #新增衍生特徵
        Feature_set['until_MS_add'] = Feature_set['until_act_mean'] + Feature_set['until_act_std']
        Feature_set['until_MS_Neg'] = Feature_set['until_act_mean'] - Feature_set['until_act_std']
        Feature_set['first_4_MS_add'] = Feature_set['first_4_act_mean'] + Feature_set['first_4_act_std']
        Feature_set['first_4_MS_Neg'] = Feature_set['first_4_act_mean'] - Feature_set['first_4_act_std']
#        Feature_set.loc[Feature_set.PART_NO.isin(is_windows['PART_NO']),'iswindow'] = 1
#        Feature_set.loc[~Feature_set.PART_NO.isin(is_windows['PART_NO']),'iswindow'] = 0

        Feature_set['Y_lable_by_week_log'] = np.log1p(Feature_set['Y_lable_by_week'])
        Feature_set_select = Feature_set[Train_feature_columns].copy()      

        actweek_dummy = pd.get_dummies(pd.Series(list(Feature_set['act_week_byyweek'])))
        actweek_dummy.columns = list(map(lambda x: 'act_week_byyweek' + str(x),actweek_dummy.columns.tolist()))
        Feature_set_select = pd.concat([Feature_set_select , actweek_dummy],axis=1)
    
        Year = Current_year
        Week = Current_week
        Current_newweek = asus_week_day[(asus_week_day['year'] == Year) & (asus_week_day['week'] == Week)].index[0]
        
        
        if state == 'current':
            asus_week_day_range = asus_week_day.loc[Current_newweek:Current_newweek]
        elif state == 'back':
            
            asus_week_day_range = asus_week_day.loc[start_newweek:Current_newweek]
        else:
            asus_week_day_range = asus_week_day.loc[start_newweek:Current_newweek]
    
    
        if (not os.path.exists(os.path.join(project_path,'predict_result/'))): 
            os.mkdir(os.path.join(project_path,'predict_result/'))

        if (not os.path.exists(os.path.join(project_path,'./predict_result/predict_result_'+Country+'/'))): 
            os.mkdir(os.path.join(project_path,'./predict_result/predict_result_'+Country+'/'))
            
        for k in range(0,len(asus_week_day_range)):
                
                Target_Year = asus_week_day_range['year'].iloc[k]
                Target_Week = asus_week_day_range['week'].iloc[k]
                Test_data = Feature_set_select[(Feature_set_select['YEAR'] == Target_Year) & (Feature_set_select['WEEK'] == Target_Week)]
                target_yweek = str(Target_Year) + '-' + str(Target_Week)
                weekly_predict_result = rolling_feature_and_predict_oneweek(Feature_set_select, Test_data, Target_Year, Target_Week)
                #畫圖看預測結果
                All_key_predict_result_df = pd.concat(weekly_predict_result)
                All_key_predict_result_df = All_key_predict_result_df.sort_values(by = ['l0name','MODEL_NAME','PART_NO','YEAR','WEEK','gapweek'])   
                All_key_predict_result_df = predict_file_format(All_key_predict_result_df)                
                All_key_predict_result_df.to_csv(project_path + '/predict_result/predict_result_'+Country+'/'+str(asus_week_day_range['year'].iloc[k])+'_w'+str(asus_week_day_range['week'].iloc[k]).zfill(2)+'.csv')

        log.info("End: {} ModelTraining".format(Country))
    log.info("End: ---------------POD_ModelTraining.py--------------------")



#                All_Train_data_week = []
#                for next_x_predict in tqdm(np.arange(safe_stock)):
#                
#                    if (Target_Week-next_x_predict - 1) <= 0:
#                        Train_until_week = Target_Week - next_x_predict - 1 + 53
#                        Train_until_year = Target_Year - 1
#                    else:
#                        Train_until_week = Target_Week - next_x_predict - 1 
#                        Train_until_year = Target_Year 
#            
#                    Train_data = Feature_set_select[((Feature_set_select['YEAR'] == Train_until_year) & (Feature_set_select['WEEK'] <= Train_until_week)) | ((Feature_set_select['YEAR'] < Train_until_year))]
#                    Train_data_week = Train_data[Train_data['gapweek'] == next_x_predict]
#                    Train_data_week = Train_data_week.sort_values(by = ['l0name', 'MODEL_NAME', 'PART_NO', 'YEAR', 'WEEK','Y_label_yweek'])
#                    Train_data_week = Train_data_week.replace([np.inf, -np.inf],1)
#                    All_Train_data_week.append(Train_data_week)
#                All_Train_data_week_df = pd.concat(All_Train_data_week)
                

