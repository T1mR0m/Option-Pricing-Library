import numpy as np

class FiniteDifferenceGreeksMixin:
    def delta(self, ds=1e-4, method="central"):
        price_0 = self.price()
        
        self.S += ds
        price_up = self.price()
        self.S -= ds
        
        if method == "forward":
            return float((price_up - price_0) / ds)
        elif method == "central":
            self.S -= ds
            price_dn = self.price()
            self.S += ds
            return float((price_up - price_dn) / (2 * ds))

    def gamma(self, ds=1e-4, method="central"):
        price_0 = self.price()
        
        self.S += ds
        price_up = self.price()
        
        if method == "forward":
            self.S += ds
            price_up_2 = self.price()
            self.S -= 2 * ds
            return float((price_up_2 - 2 * price_up + price_0) / ds**2)
        elif method == "central":
            self.S -= 2 * ds
            price_dn = self.price()
            self.S += ds
            return float((price_up - 2 * price_0 + price_dn) / ds**2)

    def vega(self, dvol=1e-2, method="central"):
        price_0 = self.price()
        
        self.sigma += dvol
        price_up = self.price()
        self.sigma -= dvol
        
        if method == "forward":
            raw_vega = (price_up - price_0) / dvol
        elif method == "central":
            self.sigma -= dvol
            price_dn = self.price()
            self.sigma += dvol
            raw_vega = (price_up - price_dn) / (2 * dvol)
            
        return float(raw_vega * 0.01)

    def theta(self, dT=1/252):
        if self.T - dT <= 0:
            return 0.0
            
        price_0 = self.price()
        
        self.T -= dT
        price_dn = self.price()
        self.T += dT
        
        raw_theta = (price_dn - price_0) / dT
        return float(raw_theta / 252)

    def rho(self, dr=1e-2, method="central"):
        price_0 = self.price()
        
        self.r += dr
        price_up = self.price()
        self.r -= dr
        
        if method == "forward":
            raw_rho = (price_up - price_0) / dr
        elif method == "central":
            self.r -= dr
            price_dn = self.price()
            self.r += dr
            raw_rho = (price_up - price_dn) / (2 * dr)
            
        return float(raw_rho * 0.01)

    def greeks(self, ds=1e-4, dvol=1e-2, dr=1e-2, dT=1/252, method="central"):
        """Returns a dictionary of all finite-difference Greeks."""
        return {
            "delta": self.delta(ds, method),
            "gamma": self.gamma(ds, method),
            "vega": self.vega(dvol, method),
            "theta": self.theta(dT),
            "rho": self.rho(dr, method)
        }

class GBMPathMixin():
    def generate_paths(self, n_steps=252, n_sims=100_000, seed=123):
        """Generates antithetic GBM price path"""

        rng = np.random.default_rng(seed=seed)
        n_paths = int(n_sims/2)
        Z_pos = rng.standard_normal((n_paths, n_steps))
        Z_neg = Z_pos * -1

        dt = self.T / n_steps
        drift = (self.r - self.q - self.sigma**2 * 0.5) * dt
        vol = np.sqrt(dt) * self.sigma 

        log_returns_pos = drift + vol * Z_pos
        log_returns_neg = drift + vol * Z_neg

        paths_pos = np.exp(np.log(self.S) + np.cumsum(log_returns_pos, axis=1))
        paths_neg = np.exp(np.log(self.S) + np.cumsum(log_returns_neg, axis=1))

        S_0_col = np.full((n_paths, 1), self.S)
        paths_pos = np.hstack([S_0_col, paths_pos])
        paths_neg = np.hstack([S_0_col, paths_neg])

        return paths_pos, paths_neg