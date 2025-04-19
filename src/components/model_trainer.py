import numpy as np
import os
import sys
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('FILEfolder','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,Train_arr,Test_arr):
        try:
            logging.info("Splitting train and test data into target")
            X_train,Y_train,X_test,Y_test = (
                Train_arr[:,:-1],
                Train_arr[:,-1],
                Test_arr[:,:-1],
                Test_arr[:,-1]
            )
            models ={
                "RandomForestRegressor":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGB":XGBRegressor(),
                "Linear": LinearRegression(),
                "CatBoost" : CatBoostRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "GradientBoost":GradientBoostingRegressor(),
                "kNN":KNeighborsRegressor()

            }
            params={
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "XGB":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear":{},
                
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "kNN":{}
                
                
            }
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models,params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
               raise CustomException("NO best Model is found")
            logging.info(f"The best  found model on training and test data set")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2_squared = r2_score(Y_test,predicted)
            return r2_squared
        except Exception as e:
            raise CustomException(e,sys)