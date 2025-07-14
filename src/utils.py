import os
import sys
import pandas as pd
from src.exception import CustomException
import numpy as np
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    '''
    This function saves the given object to a file using dill serialization.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''
    This function evaluates the models using R-squared score.
    '''
    try:
        model_report = {}

        for(model_name, model) in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)

            model_report[model_name] = r2_square
        
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    This function loads a model from the specified file path.
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)