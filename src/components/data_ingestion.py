import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import modelTrainer
from src.components.model_trainer import modelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')
                              
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingetsion method or component")
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataset as df')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('traintest split intiated')
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 11)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('train test split done and saved, returning now')

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)


        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    data_ing = DataIngestion()
    traindata, testdata = data_ing.initiate_data_ingestion()

    data_trans = DataTransformation()
    train_arr, test_arr = data_trans.initiate_data_transformation(traindata, testdata)

    modeltrain = modelTrainer()
    print(modeltrain.modelTrainer_obj(train_arr, test_arr))