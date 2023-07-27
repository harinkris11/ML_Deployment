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

            model_report:dict = eval_models(models, train_x, train_y, test_x, test_y)
            logging.info("model evaluation done")

            best_model_score = max(sorted(model_report.values()))
            best_model = max(model_report)

            if best_model_score < 0.6:
                raise CustomException("None of the models perform better")
            
            logging.info("Found the model that best fits the data")

            save_object(
                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )

            bestmodl = models[best_model]

            predicted = bestmodl.predict(test_x)
            r2_sco = r2_score(predicted, test_y)
            return r2_sco


        except Exception as e:
            raise CustomException(e, sys)

