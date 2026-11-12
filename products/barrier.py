import numpy as np
from core.base import Option
from core.mixins import FiniteDifferenceGreeksMixin, GBMPathMixin

class BarrierOption(Option, FiniteDifferenceGreeksMixin, GBMPathMixin):
    def __init__(self, S, K, B, sigma, r, q, T, opt_type, barrier_type, continuous=False):
        super().__init__(S, K, sigma, r, q, T, opt_type)

        self.B = B

        self.barrier_type = barrier_type.lower()
        if self.barrier_type not in {'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'}:
            raise ValueError("Barrier's option barrier_type must be one of 'up-and-out', 'up-and-in', 'down-and-out', or 'down-and-in'")
        
        self.continuous = continuous

    def price(self, n_steps=252, n_sims=100_000, seed=123, return_se=False):
        if self.barrier_type == 'up-and-out' and self.S > self.B:
            return 0.0
        elif self.barrier_type == "down-and-out" and self.S < self.B:
            return 0.0

        paths_pos, paths_neg = self.generate_paths(n_steps, n_sims, seed)

        dt = self.T / n_steps

        # Conitnuity correction for continuously monitored option (the defaul is discreet (daily) monitoring)
        if self.continuous:
            beta = 0.5826
            if "up" in self.barrier_type:
                B_adj = self.B * np.exp(-beta * self.sigma * np.sqrt(dt))
            elif "down" in self.barrier_type:
                B_adj = self.B * np.exp(+beta * self.sigma * np.sqrt(dt))
        else:
            B_adj = self.B

        # 1) Base payoffs based on the option type
        if self.opt_type == "call":
            base_payoff_pos = np.maximum(paths_pos[:, -1] - self.K, 0)
            base_payoff_neg = np.maximum(paths_neg[:, -1] - self.K, 0)
        elif self.opt_type == "put":
            base_payoff_pos = np.maximum(self.K - paths_pos[:, -1], 0)
            base_payoff_neg = np.maximum(self.K - paths_neg[:, -1], 0)

        # 2) Boolean arrays for indication of breaches
        if "up" in self.barrier_type:
            breaches_pos = np.any(paths_pos >= B_adj, axis=1)
            breaches_neg = np.any(paths_neg >= B_adj, axis=1)
        elif "down" in self.barrier_type:
            breaches_pos = np.any(paths_pos <= B_adj, axis=1)
            breaches_neg = np.any(paths_neg <= B_adj, axis=1)

        # 3) Set up active mask for activated options based on the crossed breach
        if "in" in self.barrier_type:
            active_pos = breaches_pos
            active_neg = breaches_neg
        elif "out" in self.barrier_type:
            active_pos = ~breaches_pos
            active_neg = ~breaches_neg

        # 4) Combine 1), 2), 3) for final payoffs and discount
        disc_factor = np.exp(-self.r * self.T)
        payoffs_pos = disc_factor * base_payoff_pos * active_pos
        payoffs_neg = disc_factor * base_payoff_neg * active_neg

        payoffs = (payoffs_pos + payoffs_neg) / 2
        price = np.mean(payoffs)

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