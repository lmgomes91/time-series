import logging

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()


def save_result_in_db(mse, mae, rmse, mape, theil_u, model_name):
    try:
        client = MongoClient(os.getenv("CONN_STRING_MONGO"))
        db = client["ppgmcs"]
        collection = db["time_series_results"]

        collection.insert_one({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "theil_u": theil_u,
            "model_name": model_name,
        })

        client.close()
    except Exception as e:
        logging.error(f"Error when try to save in DB: {e}")
        results = []


def get_results_by_model(model_name) -> pd.DataFrame:
    try:
        client = MongoClient(os.getenv("CONN_STRING_MONGO"))
        db = client["ppgmcs"]
        collection = db["time_series_results"]

        cursor = collection.find({'model_name': model_name}, {'_id': 0, 'model_name': 0})
        results = pd.DataFrame(list(cursor))

    except Exception as e:
        logging.error(f"Error when try to find documents: {e}")
        results = pd.DataFrame()

    return results
