import pandas as pd
import numpy as np
from src.models import fit_arimax, forecast_arimax
from src.utils_windows import rolling_splits

def rolling_forecast_arimax(df, order=(1,0,1)):
    """
    Perform rolling-window ARIMAX forecasts.
    
    Parameters
    ----------
    df : pd.DataFrame with ['r', 's']
    order : tuple
        ARIMA order

    Returns
    -------
    pd.DataFrame with columns:
        ['date', 'y_real', 'y_pred']
    """

    results = []

    for i, (train_mask, test_mask) in enumerate(rolling_splits(df.index)):
        train = df[train_mask]
        test = df[test_mask]

        # Skip if not enough data
        if len(train) < 30 or len(test) == 0:
            continue

        model = fit_arimax(train, order=order)
        s_next = test[["s"]].iloc[0:1]

        y_pred = forecast_arimax(model, s_next)
        y_real = test["r"].iloc[0]

        results.append({
            "date": test.index[0],
            "y_pred": y_pred,
            "y_real": y_real
        })

    return pd.DataFrame(results).set_index("date")
