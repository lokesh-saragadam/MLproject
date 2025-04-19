import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#creating a class for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('FILEfolder','Train.csv')
    test_data_path = os.path.join('FILEfolder','Test.csv')
    raw_data_path=os.path.join('FILEfolder','Data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
         df=pd.read_csv(r'notebooks\data\stud.csv')
         logging.info("Read the data as DataFrame")

         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

         logging.info("Train Test split initiated")
         Train_set,Test_set= train_test_split(df,test_size=0.2,random_state=42)
         
         Train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
         Test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
         
         logging.info("Ingestion of the data is completed")

         return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
         )

        except Exception as e:
           raise CustomException(e,sys)
           

if __name__=="__main__":
   obj = DataIngestion()
   Train_data,Test_data = obj.initiate_data_ingestion()
   data_transformation = DataTransformation()
   Train_arr,Test_arr,_=data_transformation.initiate_data_transformation(Train_data,Test_data)
   model_trainer = ModelTrainer()
   print(model_trainer.initiate_model_trainer(Train_arr=Train_arr,Test_arr=Test_arr))
