import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#Create a Dataconfig for transformation,

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("FILEfolder","Preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
      self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
       try:
         Categorical_columns =  [
            'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
         Numerical_coloumns = [
            'reading_score', 'writing_score'
         ]
         num_pipeline = Pipeline(
            steps = [
               ("imputer",SimpleImputer(strategy='median')),##handling missing values
               ("scaler",StandardScaler())##standardising the data

            ]
         )
         cat_pipeline = Pipeline(
            steps=[
               ("imputer",SimpleImputer(strategy="most_frequent")),
               ("encoder",OneHotEncoder()),
               ("scaling",StandardScaler(with_mean=False))
            ]
         )

         logging.info("Categorical and Numerical Pipelines are created")
         logging.info(f"Numerical columns: {Numerical_coloumns}")
         logging.info(f"Categorical Columns: {Categorical_columns}")   
         preprocessor = ColumnTransformer(
            [
               ("numerical_pipeline",num_pipeline,Numerical_coloumns),
               ("categorical_pipeline",cat_pipeline,Categorical_columns)
            ]
         )
         return preprocessor
       except Exception as e:
          raise CustomException(e,sys) 

    def initiate_data_transformation(self,Train_path,Test_path):
       try:
          Train_df = pd.read_csv(Train_path)
          Test_df = pd.read_csv(Test_path)

          logging.info("Train and Test data taken in")

          logging.info("Obtaining PreProcessing object")

          preprocessing_obj = self.get_data_transformer_object()
          target_column_name = "math_score"
          Numerical_coloumns = [
            'reading_score', 'writing_score'
          ]
          input_feature_train_df = Train_df.drop(columns = [target_column_name],axis=1)
          target_feature_train_df = Train_df[target_column_name]

          input_feature_test_df = Test_df.drop(columns = [target_column_name],axis=1)
          target_feature_test_df = Test_df[target_column_name]
          logging.info(f"Applying Preprocessing object on Train df and Test df")

          input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
          input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

          train_arr = np.c_[
             input_feature_train_arr,np.array(target_feature_train_df)
          ]
          test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


          save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                      obj = preprocessing_obj)
          logging.info(f"Saved Preprocessing object")
          return (
             train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path,
          )
       except Exception as e:
          raise CustomException(e,sys)       

