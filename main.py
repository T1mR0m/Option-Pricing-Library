import numpy as np
from datetime import datetime

from market_data.data_loader import MarketDataLoader
from market_data.vol_estimators import VolatilityEstimator
from vol_calibration.implied_vol import ImpliedVolCalibrator

from products.european import EuropeanOption
from products.american import AmericanOption
from products.asian import AsianOption
from products.autocallable import Autocallable
from products.barrier import BarrierOption
from products.binary import BinaryOption

def main():

    opt_style_dict = {
        "european": EuropeanOption,
        "american": AmericanOption,
        "asian": AsianOption,
        "barrier": BarrierOption,
        "binary": BinaryOption,
        "autocallable": Autocallable
    }

    ticker = input('Enter the underlying ticker (default AAPL): ').upper() or 'AAPL'
    print(f'Fetching data for {ticker}')

    loader = MarketDataLoader(ticker)

    spot = loader.get_spot_price()
    q = loader.get_div_yield()
    r = loader.get_risk_free_rate()
    returns = loader.get_log_returns()

    while True:
        try:
            opt_style = input("Enter option style (e.g., European, American, Asian, etc): ").strip().lower()
            if opt_style not in opt_style_dict:
                raise ValueError
            break
        except ValueError:
            print('Enter correct option style among [european, american, asian, autocallable, barrier, or binary]')

    opt_type = input("Enter the option's type (call or put): ").strip().lower()
    expiration_date = input("Enter the option's expiration date (YYYY-MM-DD): ").strip()
    target_strike = float(input("Enter the option's target strike: "))

    print(f'\nSpot: {spot:.2f}, Dividend Yield: {q:.4f}, Risk-free Rate: {r:.4f}')

    vol_estimator = VolatilityEstimator(returns)
    garch_sigma = vol_estimator.calc_garch_volatility()
    print(f'GARCH(1, 1) vol: {garch_sigma:.4f}')

    real_option_data, target_date = loader.get_closest_option(expiration_date, opt_type, target_strike)

    today = loader.get_latest_market_date()
    target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    # time_to_expiry = float(np.busday_count(today, target_date) / 252.0) # 252 convention
    time_to_expiry = float((target_date - today).days / 365.0) # 365 convention

    live_iv = real_option_data['impliedVolatility']

    bid, ask = real_option_data['bid'], real_option_data['ask']
    if bid > 0 and ask > 0:
        market_price = (bid + ask) / 2
    else:
        market_price = real_option_data['lastPrice']

    actual_strike = real_option_data['strike']

    print(f"\n--- Live Market Data for Strike {actual_strike} ---")

    print(f"Latest Bid: {bid}")
    print(f"Latest Ask: {ask}")
    print(f"Last Trade Date: {real_option_data['lastTradeDate']}")

    print(f"Market Price: {market_price}")
    print(f"Quoted IV: {live_iv:.4f}")

    option = opt_style_dict[opt_style](
        S=spot, K=actual_strike, 
        sigma=garch_sigma, r=r, q=q, 
        T=time_to_expiry, opt_type=opt_type
    )

    model_price = option.price()
    print(f"Model Price: {model_price:.4f}")

    calibrator = ImpliedVolCalibrator(option, target_price=market_price)
    calculated_iv = calibrator.calc_iv_newton_raphson(initial_vol=garch_sigma)

    print(f"Calculated IV from Newton-Raphson: {calculated_iv:.4f}")

if __name__ == "__main__":
    main()