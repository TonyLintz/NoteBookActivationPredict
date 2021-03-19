# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:36:31 2019

@author: user
"""

#Program Control
import os 

dirname, filename = os.path.split(os.path.abspath(__file__)) 
os.path.split(dirname)
workspace = os.path.split(dirname)[0]

project_state = 'Current'
#step1.Datapreparation
os.system('python '+dirname + '\\POD_DataPreparation.py '+workspace+'')


#step2.ModelTraining and Predict
os.system('python '+dirname + '\\POD_ModelTraining.py '+workspace+' '+project_state+'')



#step3.switch
os.system('python '+dirname + '\\switch_vol1.py '+workspace+'')


#step4.concat all country
os.system('python '+dirname + '\\Concat_country_result.py '+workspace+'')

#step5.fill act
os.system('python '+dirname + '\\fill_act.py '+workspace+'')
