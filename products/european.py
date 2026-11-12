import numpy as np
from scipy.stats import norm
from core.base import Option
from core.mixins import GBMPathMixin, FiniteDifferenceGreeksMixin

class EuropeanOption(Option, GBMPathMixin, FiniteDifferenceGreeksMixin):
    def d1_d2(self):
        d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price_analytical(self):
        d1, d2 = self.d1_d2()

        if self.opt_type == "call":
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.opt_type == "put":
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return float(price)

    def greeks_analytical(self):
        d1, d2 = self.d1_d2()

        gamma = (norm.pdf(d1) * np.exp(-self.q * self.T)) / (self.S * self.sigma * np.sqrt(self.T)) 
        raw_vega = self.S * np.sqrt(self.T) * norm.pdf(d1) * np.exp(-self.q * self.T)
        # Dividing by 100 to represent the measure per single unit of vol (sensitivity to 1% increase in vol)
        quoted_vega = raw_vega / 100 

        if self.opt_type == "call":
                delta = np.exp(-self.q * self.T) * norm.cdf(d1) 
        
                raw_theta = (
                        -((self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T))
                        / (2 * np.sqrt(self.T))) 
                        + self.q * self.S * norm.cdf(d1) * np.exp(-self.q * self.T) 
                        - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                )
                quoted_theta = raw_theta / 252 # convert to time decay per day
        
                raw_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
                quoted_rho = raw_rho / 100 # Dividing by 100 to represent the measure per 1 % move in int rates
        
        elif self.opt_type == "put":
            delta = np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
    
            raw_theta = (
                    -((self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) 
                    / (2 * np.sqrt(self.T))) 
                    - self.q * self.S * norm.cdf(-d1) * np.exp(-self.q * self.T) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            )
            quoted_theta = raw_theta / 252
            
            raw_rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
            quoted_rho = raw_rho / 100
    
        else:
            raise ValueError("Option type (opt_type) should be either 'call' or 'put'")
    
        greeks = {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(quoted_vega),
            "theta": float(quoted_theta),
            "rho": float(quoted_rho)
        }
    
        return greeks

    def price_mc(self, n_steps=1, n_sims=100_000, seed=123, return_se = False):
        paths_pos, paths_neg = self.generate_paths(n_steps, n_sims, seed)

        if self.opt_type == "call":
            payoffs_pos = np.exp(-self.r * self.T) * np.maximum(paths_pos[:, -1] - self.K, 0)
            payoffs_neg = np.exp(-self.r * self.T) * np.maximum(paths_neg[:, -1] - self.K, 0)
        elif self.opt_type == "put":
            payoffs_pos = np.exp(-self.r * self.T) * np.maximum(self.K - paths_pos[:, -1], 0)
            payoffs_neg = np.exp(-self.r * self.T) * np.maximum(self.K - paths_neg[:, -1], 0)

        payoffs = (payoffs_pos + payoffs_neg) / 2
        price = np.mean(payoffs)

        if return_se:
            se = np.std(payoffs, ddof=1) / np.sqrt(int(n_sims / 2))
            return float(price), float(se)

        return float(price)

    def price(self):
        return self.price_analytical()

    def delta(self):
        return self.greeks_analytical()['delta']

    def gamma(self):
        return self.greeks_analytical()['gamma']
        
    def vega(self):
        return self.greeks_analytical()['vega']

    def rho(self):
        return self.greeks_analytical()['rho']

    def theta(self):
        return self.greeks_analytical()['theta']

    def greeks(self):
        return self.greeks_analytical()