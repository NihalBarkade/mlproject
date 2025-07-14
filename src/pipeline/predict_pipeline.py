import sys 
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        ''' This method predicts the output using the trained model and preprocessor.
        It expects features to be a pandas DataFrame. '''

        logging.info("Starting prediction pipeline")
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            logging.info("Prediction completed successfully")
            print(f"Predictions: {preds}")
            return preds
        
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        ''' This method converts the custom data input into a pandas DataFrame. '''
        logging.info("Converting custom data input to DataFrame")
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            df =  pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data input converted to dataframe")

            return df
        
        except Exception as e:
            logging.error("Error occurred while converting custom data to DataFrame")
            raise CustomException(e, sys)