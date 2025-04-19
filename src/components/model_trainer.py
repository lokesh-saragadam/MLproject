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
    trained_model_file_path = os.path.join('FILEfolder','preprocessor.pkl')
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
                "GraduentBoost":GradientBoostingRegressor(),
                "kNN":KNeighborsRegressor()

            }

            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models)

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