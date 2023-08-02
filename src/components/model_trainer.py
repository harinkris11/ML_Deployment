import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import eval_models

@dataclass
class modelTrainerConfig():
    model_trainer_file_path = os.path.join('artifacts','model.pkl')

class modelTrainer():
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()
    
    def modelTrainer_obj(self, traindata, testdata):
        try:

            train_x, train_y, test_x, test_y =(
                traindata[:,:-1], traindata[:,-1],
                testdata[:,:-1], testdata[:,-1]
            )
            logging.info("train and test - x, y split done")

            models = {
                "Categoriocal Boost": CatBoostRegressor(verbose = True),
                "XG Boost": XGBRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boosting": AdaBoostRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K Nearest Neighbours": KNeighborsRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K Nearest Neighbours":{},
                "XG Boost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Categoriocal Boost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Ada Boosting":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = eval_models(models, train_x, train_y, test_x, test_y, params = params)
            logging.info("model evaluation done")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = max(model_report)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("None of the models perform better")
            
            logging.info("Found the model that best fits the data")

            save_object(
                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )


            predicted = best_model.predict(test_x)
            r2_sco = r2_score(predicted, test_y)
            return r2_sco


        except Exception as e:
            raise CustomException(e, sys)

