from abc import ABC, abstractmethod

class Option(ABC):
    def __init__(self, S, K, sigma, r, q, T, opt_type):
        self.S = S
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.T = T
        self.opt_type = opt_type.lower()

        self._validate()

    def _validate(self):
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be greater than zero")
        if self.sigma <= 0:
            raise ValueError("sigma (volatility) must be greater than zero")
        if self.T <= 0:
            raise ValueError("T (time to option's expiry) must be greater than zero")
        if self.opt_type not in {"call", "put"}:
            raise ValueError("opt_type (option type) must be either 'call' or 'put'")

    @abstractmethod
    def price(self):
        """Calculate and return option price"""

    @abstractmethod
    def greeks(self):
        """Calculate and return the option Greeks"""