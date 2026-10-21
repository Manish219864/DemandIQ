import os
import sys
# pyrefly: ignore[missing-import]
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    validation_data_path: str = os.path.join("artifacts", "validation.csv")
    kaggle_test_path: str = os.path.join("artifacts", "Kaggle_Test.csv")
    transactions_path: str = os.path.join("artifacts", "transactions.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Starting Demand IQ data ingestion process")
        try:
            data_dir = os.path.join("notebook", "Data")
            datasets = {
                "train": pd.read_csv(os.path.join(data_dir, "train.csv")),
                "test": pd.read_csv(os.path.join(data_dir, "test.csv")),
                "stores": pd.read_csv(os.path.join(data_dir, "stores.csv")),
                "oil": pd.read_csv(os.path.join(data_dir, "oil.csv")),
                "holidays": pd.read_csv(os.path.join(data_dir, "holidays_events.csv")),
                "transactions": pd.read_csv(os.path.join(data_dir, "transactions.csv")),
                "sample_submission": pd.read_csv(os.path.join(data_dir, "sample_submission.csv")),
            }
            
            logging.info("All seven CSV datasets loaded successfully")

            # Extract individual DataFrames
            df = datasets["train"].copy()
            stores_df = datasets["stores"]
            oil_df = datasets["oil"]
            holidays_df = datasets["holidays"]
            transactions_df = datasets["transactions"]

            # Convert date columns to datetime
            for data in [df, oil_df, holidays_df, transactions_df]:
                data["date"] = pd.to_datetime(data["date"])

            # Merge store information
            df = df.merge(stores_df, on="store_nbr", how="left")

            # Merge oil prices
            df = df.merge(oil_df, on="date", how="left")

            # Merge transaction information
            df = df.merge(
                transactions_df,
                on=["date", "store_nbr"],
                how="left"
            )

            print("Merged DataFrame shape:", df.shape)
            print(df.head())
          
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
          
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
          
            logging.info("Chronological train-validation split initiated")

            # Ensure date is in datetime format
            df["date"] = pd.to_datetime(df["date"])

            # Sort records by date
            df = df.sort_values("date")

            # Find the date at which validation begins
            unique_dates = sorted(df["date"].unique())
            split_index = int(len(unique_dates) * 0.8)
            validation_start = unique_dates[split_index]

            # Earlier dates -> training data
            train_set = df[df["date"] < validation_start].copy()

            # Later dates -> validation data
            validation_set = df[df["date"] >= validation_start].copy()

            logging.info(f"Training data shape: {train_set.shape}")
            logging.info(f"Validation data shape: {validation_set.shape}")
            logging.info(f"Validation starts from: {validation_start}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
          
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)   