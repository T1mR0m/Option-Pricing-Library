import numpy as np
import yfinance as yf
from datetime import datetime

class MarketDataLoader():
    def __init__(self, ticker):
        self.ticker = ticker
        self.yfticker = yf.Ticker(ticker)
        self.data = None

    def fetch_historical_data(self, period="1y"):
        """
        Fetches historical daily OHLCV data.
        Calculates daily log returns for further volatility modeling.
        """
        df = self.yfticker.history(period=period)

        if df.empty:
            raise ValueError(f"No market data found for {self.ticker}")

        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        self.data = df

        return self.data

    def get_spot_price(self):
        if self.data is None:
            self.fetch_historical_data()

        spot = float(self.data['Close'].iloc[-1])

        return spot

    def get_log_returns(self):
        if self.data is None:
            self.fetch_historical_data()

        log_returns = self.data['log_return'].dropna()

        return log_returns

    def get_risk_free_rate(self):
        """
        Approximates the risk-free rate using the 13-week Treasury Bill (^IRX).
        Returns the decimal rate (e.g., 0.052 for 5.2%).
        """
        irx = yf.Ticker("^IRX")
        df_irx = irx.history(period="5d")

        if df_irx.empty:
            print("Failed to retrieve T-Bill data from YahooFinance.")

            while True:
                try:
                    raw_input = input("Please provide the risk-free rate as a decimal (e.g., 0.05 for 5%): ")
                    clean_input = raw_input.replace(',', '.')

                    risk_free_rate = float(clean_input)

                    if abs(risk_free_rate) >= 1.0:
                        print(f"Note: You entered {risk_free_rate}. Assuming you meant {risk_free_rate}%. Auto-converting to {risk_free_rate / 100.0}")
                        risk_free_rate = risk_free_rate / 100.0

                    return risk_free_rate
                
                except ValueError:
                    print("Invalid input. Please enter a decimal number like 0.05.")

        latest_irx = float(df_irx['Close'].iloc[-1]) * 0.01

        return latest_irx

    def get_option_chain(self, expiration_date, option_type):
        """Fetches the full options DataFrame for a given date and type."""
        option_dates = self.yfticker.options

        if expiration_date in option_dates:
            target_date = expiration_date
        else:
            expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')
            target_date = min(option_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - expiration_datetime))

        chain_data = self.yfticker.option_chain(target_date)

        if option_type == "call":
            return chain_data.calls, target_date
        elif option_type == "put":
            return chain_data.puts, target_date
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def get_closest_option(self, expiration_date, option_type, target_strike):
        """Filters the chain to return the single row closest to the target strike."""
        df_chain, target_date = self.get_option_chain(expiration_date, option_type)

        closest_strike_idx = (df_chain['strike'] - target_strike).abs().idxmin()
        return df_chain.loc[closest_strike_idx], target_date

    def get_latest_market_date(self):
        if self.data is None:
            self.fetch_historical_data()
        return self.data.index[-1].date()

    # def get_div_yield(self):
    #     q = self.yfticker.info.get("dividendYield")
    #     return q if q is not None else 0 

    def get_div_yield(self):
        info = self.yfticker.info
        annual_dividend = info.get("dividendRate")
        share_price = info.get("currentPrice")

        calculatedYield = annual_dividend / share_price

        return calculatedYield