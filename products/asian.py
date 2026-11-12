import numpy as np
from core.base import Option
from core.mixins import FiniteDifferenceGreeksMixin, GBMPathMixin

class AsianOption(Option, FiniteDifferenceGreeksMixin, GBMPathMixin):
    def __init__(self, S, K, sigma, r, q, T, opt_type, average_type="arithmetic"):
        super().__init__(S, K, sigma, r, q, T, opt_type)

        self.average_type = average_type.lower()
        if self.average_type not in {"arithmetic", "geometric"}:
            raise ValueError("average_type must be either 'arithmetic' or 'geometric'")

    def price(self, n_steps=252, n_sims=100_000, seed=123, return_se=False):
        paths_pos, paths_neg = self.generate_paths(n_steps, n_sims, seed)

        if self.average_type == "arithmetic":
            averages_pos = np.mean(paths_pos, axis=1)
            averages_neg = np.mean(paths_neg, axis=1)
        elif self.average_type == "geometric":
            averages_pos = np.exp(np.mean(np.log(paths_pos), axis=1))
            averages_neg = np.exp(np.mean(np.log(paths_neg), axis=1))

        if self.opt_type == "call":
            payoffs_pos = np.maximum(averages_pos - self.K, 0)
            payoffs_neg = np.maximum(averages_neg - self.K, 0)
        elif self.opt_type == "put":
            payoffs_pos = np.maximum(self.K - averages_pos, 0)
            payoffs_neg = np.maximum(self.K - averages_neg, 0)

        payoffs = (payoffs_pos + payoffs_neg) / 2
        price = np.exp(-self.r * self.T) * np.mean(payoffs)

        if return_se:
            se = np.std(payoffs, ddof=1) / np.sqrt(int(n_sims / 2))
            return float(price), float(se)
        
        return float(price)

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