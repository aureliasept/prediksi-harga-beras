import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R²": r2_score(y_true, y_pred)
    }


def compare_models(y_true, pred_rf, pred_svr):
    rf = calculate_metrics(y_true, pred_rf)
    svr = calculate_metrics(y_true, pred_svr)

    return pd.DataFrame([
        {
            "Model": "Random Forest",
            **rf
        },
        {
            "Model": "SVR",
            **svr
        }
    ])