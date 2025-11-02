import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch.univariate import EGARCH, ConstantMean

def fit_arimax(train_df, order=(1,0,1)):
    model = ARIMA(train_df['r'], exog=train_df[['s']], order=order)
    res = model.fit(method='innovations_mle')
    return res

def fit_egarch(residuals, s):
    am = ConstantMean(residuals)
    eg = EGARCH(p=1, q=1)
    am.volatility = eg
    am.volatility._exog = np.asarray(s).reshape(-1,1)
    result = am.fit(disp="off")
    return result

def forecast_arimax(model_res, s_next):
    f = model_res.get_forecast(steps=1, exog=s_next)
    return float(f.predicted_mean)

def forecast_egarch(vol_res):
    f = vol_res.forecast(horizon=1)
    return float(f.variance.iloc[-1, 0])
