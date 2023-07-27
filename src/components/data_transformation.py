import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    '''Function used for different types of data'''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformer_obj(self):
        try:
            num_features = ["writing score","reading score"]
            cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Num and Cat encoding done")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipe", num_pipeline, num_features),
                    ("cat_pipe", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed, getting preprocessing obj")

            preprocess_obj = self.get_transformer_obj()

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocess_obj
            )

            target = "math score"
            num_features = ['writing score', 'reading score']

            input_features_train = train_df.drop(columns = [target], axis = 1 )
            target_train = train_df[target]

            input_features_test = test_df.drop(columns = [target], axis = 1)
            target_test = test_df[target]

            logging.info(f"Preprocessing on Train and test df")

            input_feature_train_Arr = preprocess_obj.fit_transform(input_features_train)
            input_feature_test_Arr = preprocess_obj.transform(input_features_test)

            train_arr = np.c_[input_feature_train_Arr, np.array(target_train)]
            test_arr = np.c_[input_feature_test_Arr, np.array(target_test)]

            logging.info("data has been preprocessed")

            return(train_arr, test_arr)


        except Exception as e:
            raise CustomException(e, sys)