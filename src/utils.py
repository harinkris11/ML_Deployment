import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def eval_models(models, x_train, y_train, x_test, y_test):
    try:
        res = {}

        for model in models:
            train_model = models[model]
            train_model.fit(x_train, y_train)

            y_train_pred = train_model.predict(x_train)
            y_pred = train_model.predict(x_test)

            tr_score = r2_score(y_train_pred, y_train)
            score = r2_score(y_pred, y_test)

            res[model] = score
        
        return res
    
    except Exception as e:
        raise CustomException(e, sys)
