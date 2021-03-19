## NB Auto Alert ##

### Project 維護規範 ###

1. 修改程式請從 master 建立 branch 後開始修改 (Develop-YourName-Purpose)
2. 請將程式放在 Code 目錄下，再根據 module 分類
3. 請將中繼資料或是輸出的資料放在 Data 目錄下，此資料夾下的資料原則上不放到 git lab
4. 請將資料源放在 git lab 那台 share storage
5. 程式需分成主程式負責流程 (單一程式py檔) 與其他功能程式 (以function包裝的python程式)，切勿重頭到尾寫在一個程式裡
6. 路徑等會因不同人維護而有不同設定的變數，請透過 config 方式統一管理
7. 請補充該 Module 對應的 README.md，說明程式進入點 & 執行須知
8. 請將預測路徑中的.gitkeep刪除，以免程式誤讀


Code 執行流程:

Main_Code:
1. POD_DataPreparation.py
2. get_act_data.py
3. POD_ModelTraining.py
4. link_two_side.py
5. switch_vol1.py
6. Concat_country_result.py
7. fill_act.py

func:
1.Calculate.py
2.Create_feature_func.py
3.Processing.py

============================

Eid_Event_Main_Code:
1. EidDataPreparation.py
2. EidModelRegression.py
3. CorrectEventPredict.py

func:
1.EidUsingFuctionSet.py
2.IDEvent.py


============================
Project 使用說明

1. 修改設定檔:
	a. 先複製一份/Main_Code/Config.Sample.py，並改名為Config.py，此檔不需被git 管理
	b. 修改User_path 、User_NB_path的路徑
	
2. console下的執行方式為
	a. python "主程式" "TarketWeek"(格式範例:2020-33)
	b.範例: python /opt3/home/Tony/POD_Model/Eid_Event_Main_Code/EidModelRegression.py 2020-07
	

============================
中繼檔路徑:
1. POD_DataPreparation輸出 在 '.Data/'
2. get_act_Dtaa輸出 在 './act_Data/'
3. POD_ModelTraining輸出 在 './predict_result/' (程式裡自薦路徑)
4. link_two_side 輸出在 './predict_result/' (POD_ModelTraining裡自薦路徑)
5. switch_vol1 輸出在 './Switch/' 


