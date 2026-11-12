import numpy as np
from core.base import Option
from core.mixins import FiniteDifferenceGreeksMixin

class AmericanOption(Option, FiniteDifferenceGreeksMixin):
    def price(self, n_steps=500):
        dt = self.T / n_steps

        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp((self.r - self.q) * dt)
        p = (a - d) / (u - d)

        # Initialize down moves array
        i = np.arange(n_steps + 1)

        # Calculate terminal stock prices using down moves array
        terminal_stock_prices = self.S * u**(n_steps - i) * d**(i)

        # Calculate option_values array for terminal option values based on the option type

        if self.opt_type == "call":
            option_values = np.maximum(terminal_stock_prices - self.K, 0)
        elif self.opt_type == "put":
            option_values = np.maximum(self.K - terminal_stock_prices, 0)

        # Recursively compute option value at T = 0
        for j in range(n_steps - 1, -1, -1):
            i = np.arange(j + 1)
            current_stock_price = self.S * u**(j - i) * d**(i)

            continuation_values = np.exp(-self.r * dt) * (p * option_values[:-1] + (1-p) * option_values[1:])

            if self.opt_type == "call":
                exercise_values = np.maximum(current_stock_price - self.K, 0)
            elif self.opt_type == "put":
                exercise_values = np.maximum(self.K - current_stock_price, 0)

            # update option_values
            option_values = np.maximum(continuation_values, exercise_values)

        return float(option_values[0])

    def delta(self, ds=1e-4, method="central"):
        return super().delta(ds, method)

    def gamma(self, ds=1e-4, method="central"):
        return super().gamma(ds, method)

    def vega(self, dvol=1e-2, method="central"):
        return super().vega(dvol, method)

    def theta(self, dT=1/252):
        return super().theta(dT)

    def rho(self, dr=1e-2, method="central"):
        return super().rho(dr, method)

    def greeks(self, ds=1e-4, dvol=1e-2, dr=1e-2, dT=1/252, method="central"):
        return super().greeks(ds, dvol, dr, dT, method)