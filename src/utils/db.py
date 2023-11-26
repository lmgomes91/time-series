from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()


def save_result_in_db(mse, mae, rmse, mape, theil_u, model_name):
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
