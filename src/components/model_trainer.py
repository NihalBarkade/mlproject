import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info('Model Trainer Initiated')
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info('Data split into training and testing sets')

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            logging.info('Models initialized for training')

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            logging.info('Model evaluation completed')

            best_model_score = max(sorted(model_report.values()))

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info("Saving the best model")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"Best model: {best_model_name} with R-squared score: {best_model_score}")
            
            logging.info('Model training and saving completed successfully')
            return best_model_name, best_model_score

        except Exception as e:
            logging.error("An error occurred during model training")
            raise CustomException(e, sys)