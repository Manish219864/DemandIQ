import sys
import numpy as np
import pandas as pd
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            df = features.copy()

            # Feature engineering: extract date components matching model training
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["month"] = df["date"].dt.month
                df["day"] = df["date"].dt.day
                df["dayofweek"] = df["date"].dt.dayofweek
                df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

            if "type" in df.columns and "store_type" not in df.columns:
                df["store_type"] = df["type"]

            if "oil_price" in df.columns and "dcoilwtico" not in df.columns:
                df["dcoilwtico"] = df["oil_price"]

            if "holiday_type" not in df.columns:
                df["holiday_type"] = "None"

            if "transactions" not in df.columns:
                df["transactions"] = 0

            # Transform categoricals with trained OrdinalEncoder
            cat_cols = ["family", "city", "state", "store_type", "holiday_type"]
            df[cat_cols] = preprocessor.transform(df[cat_cols])

            # Align feature columns expected by XGBRegressor model
            model_features = getattr(model, "feature_names_in_", None)
            if model_features is not None:
                df = df[model_features]

            predictions = model.predict(df)
            predictions = np.clip(predictions, 0, None)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        store_nbr: int,
        family: str,
        date: str,
        onpromotion: int,
        oil_price: float,
        type: str,
        city: str,
        state: str,
        cluster: int,
    ):
        self.store_nbr = int(store_nbr)
        self.family = str(family)
        self.date = str(date)
        self.onpromotion = int(onpromotion)
        self.oil_price = float(oil_price)
        self.type = str(type)
        self.city = str(city)
        self.state = str(state)
        self.cluster = int(cluster)

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "store_nbr": [self.store_nbr],
                "family": [self.family],
                "date": [self.date],
                "onpromotion": [self.onpromotion],
                "oil_price": [self.oil_price],
                "type": [self.type],
                "city": [self.city],
                "state": [self.state],
                "cluster": [self.cluster],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

    get_data_as_data_frame = get_data_as_dataframe
