import numpy as np
from core.base import Option
from core.mixins import FiniteDifferenceGreeksMixin, GBMPathMixin

class Autocallable(Option, FiniteDifferenceGreeksMixin, GBMPathMixin):
    def __init__(self, S, K, sigma, r, q, T, opt_type, 
                 autocallable_barrier, protection_barrier, coupon_rate, obs_indices=[252,504,756]):
        super().__init__(S, K, sigma, r, q, T, opt_type)

        self.autocallable_barrier = autocallable_barrier
        self.protection_barrier = protection_barrier
        self.coupon_rate = coupon_rate
        self.obs_indices = obs_indices

    def price(self, n_steps=756, n_sims=100_000, seed=123, return_se=False):
        paths_pos, paths_neg = self.generate_paths(n_steps, n_sims, seed)

        paths_all = np.vstack([paths_pos, paths_neg])

        payoffs = np.zeros(n_sims)

        for i in range(n_sims):
            path = paths_all[i]
            autocalled = False

            for year, idx in enumerate (self.obs_indices):
                act_year = year + 1

                if path[idx] >= self.autocallable_barrier:
                    payoff = np.exp(-self.r * act_year) * self.S * (1 + self.coupon_rate * act_year)
                    payoffs[i] = payoff
                    autocalled = True

                    break

            if not autocalled:
                final_price = path[-1]

                if final_price >= self.protection_barrier:
                    payoffs[i] = np.exp(-self.r * self.T) * self.S
                elif final_price < self.protection_barrier:
                    payoffs[i] = np.exp(-self.r * self.T) * final_price

        half = int(n_sims / 2)
        paired_payoffs = (payoffs[:half] + payoffs[half:]) / 2

        price = np.mean(paired_payoffs)

        if return_se:
            se = np.std(paired_payoffs) / np.sqrt(half)
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