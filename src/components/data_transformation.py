import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline.
        '''

        logging.info("Data Transformation Object Creation initiated")
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info("Numerical columns standardization pipeline created")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Column transformer created with numerical and categorical pipelines")

            return preprocessor
        
        except Exception as e:
            logging.error("An error occurred during data transformation object creation")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation process by loading the train and test datasets,
        applying the preprocessing steps, and saving the preprocessor object.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'math_score'

            # Splitting the features and target variable from the training and testing dataframes
            logging.info("Splitting features and target variable")
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Applying the preprocessing object to the training and testing dataframes
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combining the transformed features with the target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Preprocessing object saved successfully")

            logging.info("Data transformation completed successfully")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            logging.error("An error occurred during data transformation initiation")
            raise CustomException(e, sys)