from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()


def save_result_in_db(mse, mae, rmse, mape, theil_u, model_name):
    client = MongoClient(os.getenv("CONN_STRING_MONGO"))
    db = client["ppgmcs"]
    collection = db["time_series_results"]

    mse = np.array(mse)
    mae = np.array(mae)
    rmse = np.array(rmse)
    mape = np.array(mape)
    theil_u = np.array(theil_u)

    collection.insert_one({
        "mse": mse.tolist(),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
        "mape": mape.tolist(),
        "theil_u": theil_u.tolist(),
        "model_name": model_name,
    })

    client.close()
