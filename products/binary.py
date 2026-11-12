import numpy as np
from core.base import Option
from core.mixins import GBMPathMixin, FiniteDifferenceGreeksMixin

class BinaryOption(Option, GBMPathMixin, FiniteDifferenceGreeksMixin):
    def __init__(self, S, K, sigma, r, q, T, opt_type, Q):
        super().__init__(S, K, sigma, r, q, T, opt_type)

        self.Q = Q

    def price_mc(self, n_steps=1, n_sims=100_000, seed=123, return_se=False):
        paths_pos, paths_neg = self.generate_paths(n_steps, n_sims, seed)
        paths_all = np.vstack([paths_pos, paths_neg])
        terminal_stock_prices = paths_all[:, -1]

        if self.opt_type == "call":
            payoffs = np.exp(-self.r * self.T) * np.where(terminal_stock_prices > self.K, self.Q, 0)
        elif self.opt_type == "put":
            payoffs = np.exp(-self.r * self.T) * np.where(terminal_stock_prices < self.K, self.Q, 0)

        half = int(n_sims / 2)
        paired_payoffs = (payoffs[half:] + payoffs[:half]) / 2

        price = np.mean(paired_payoffs)

        if return_se:
            se = np.std(paired_payoffs, ddof=1) / np.sqrt(half)
            return float(price), float(se)

        return float(price)

    def price_analytical(self):
        from scipy.stats import norm
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1  - self.sigma * np.sqrt(self.T)

        if self.opt_type == "call":
            price = self.Q * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.opt_type == "put":
            price = self.Q * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return float(price)

    def price(self):
        return self.price_analytical()

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