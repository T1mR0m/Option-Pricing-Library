import numpy as np
import pandas as pd
from datetime import datetime

from market_data.data_loader import MarketDataLoader
from vol_calibration.implied_vol import ImpliedVolCalibrator

class VolatilitySurface():
    def __init__(self, data_loader: MarketDataLoader, option_class, moneyness_range=(0.8, 1.2), min_volume=1, min_maturity = 7/365):
        self.loader = data_loader
        self.option_class = option_class

        self.moneyness_range = moneyness_range
        self.min_volume = min_volume
        self.min_maturity = min_maturity

        self.surface_data = [] # Strike, Maturity, IV


    def build_surface(self):
        surface_data = []

        S = self.loader.get_spot_price()
        r = self.loader.get_risk_free_rate()
        q = self.loader.get_div_yield()

        initial_vol_guess = 0.2

        today = self.loader.get_latest_market_date()

        expiration_dates = self.loader.yfticker.options

        for date_str in expiration_dates:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            time_to_expiry = float((date - today).days / 365.0)
            if time_to_expiry < self.min_maturity:
                continue

            chain = self.loader.yfticker.option_chain(date_str)
            calls = chain.calls
            puts = chain.puts

            OTM_calls = calls[calls['strike'] > S].copy()
            OTM_puts = puts[puts['strike'] <= S].copy()

            OTM_calls['opt_type'] = "call"
            OTM_puts['opt_type'] = "put"

            OTM_options = pd.concat([OTM_calls, OTM_puts], ignore_index=True)

            lower_bound = S * self.moneyness_range[0]
            upper_bound = S * self.moneyness_range[1]

            valid_options = OTM_options[
                (OTM_options['volume'] >= self.min_volume) &
                (OTM_options['strike'] >= lower_bound) &
                (OTM_options['strike'] <= upper_bound)
            ].copy()
            
            for row in valid_options.itertuples():
                strike, bid, ask, opt_type = row.strike, row.bid, row.ask, row.opt_type

                if bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                else:
                    market_price = row.lastPrice

                option = self.option_class(
                    S=S, K=strike, 
                    sigma=initial_vol_guess,
                    r=r, q=q, 
                    T=time_to_expiry, opt_type=opt_type
                    )

                if market_price > 0.01:
                    iv_calibrator = ImpliedVolCalibrator(option, market_price)
                else:
                    continue

                try:
                    iv = iv_calibrator.calc_iv_newton_raphson()
                    if iv is None or np.isnan(iv) or iv <= 0 or iv > 5.0:
                        continue
                except (ValueError, ZeroDivisionError, RuntimeError):
                    continue

                row_coordinate = {
                    "Strike": strike,
                    "Moneyness": strike / S,
                    "Log Moneyness": np.log(strike / S),
                    "Expiration Date": date_str,
                    "Time to Maturity": time_to_expiry,
                    "Option Type": opt_type,
                    "Market Price": market_price,
                    "Calibrated IV": iv
                }

                surface_data.append(row_coordinate)

        self.surface_data = surface_data

        vol_surface = pd.DataFrame(self.surface_data)

        return vol_surface