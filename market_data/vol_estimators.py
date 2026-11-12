import numpy as np
import pandas as pd
from arch import arch_model

class VolatilityEstimator():
    def __init__(self, log_returns):
        self.log_returns = log_returns

    def calc_historical_volatility(self, window=30):
        """
        Calculates the annualized historical volatility over the given window.
        """
        vol = self.log_returns.tail(window).std()
        vol_annualized = vol * np.sqrt(252)

        return float(vol_annualized)

    def calc_garch_volatility(self):
        """
        Fits a GARCH(1,1) model and extracts the annualized conditional 
        volatility forecast for the next trading day.
        """
        # Scale log returns in order to safeguard against failure of the MLE of the GARCH model to converge 
        # due to small values of daily log returns
        scaled_log_returns = self.log_returns * 100

        garch = arch_model(scaled_log_returns, vol="GARCH", p=1, q=1).fit(disp="off")
        garch_forecast = garch.forecast(horizon=1)
        variance_forecast = float(garch_forecast.variance.iloc[-1,0])
        daily_vol_forecast = np.sqrt(variance_forecast)
        vol_annualized = (daily_vol_forecast / 100) * np.sqrt(252)

        return vol_annualized