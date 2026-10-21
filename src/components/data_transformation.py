import sys
from dataclasses import dataclass

import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.pipeline.logger import logging
from src.pipeline.exception import CustomException

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl") # form preprocessor.pkl file in artifacts folder

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            numerical_columns = ["onpromotion", "oil_price"]
            categorical_columns = ["store_nbr", "family", "type", "city", "state", "cluster"]

            num_pipeline = Pipeline(
                steps=[

                    ("imputer", SimpleImputer(strategy="median")),

                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),

                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),

                    ("scaler", StandardScaler(with_mean=False))

                ]
            )

            logging.info("Numerical and Categorical columns are processed")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            validation_df = pd.read_csv(validation_path)

            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "sales"
            drop_columns = [target_column_name, "date"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = validation_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = validation_df[target_column_name]

            logging.info("Applying preprocessing object on training and validation datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

