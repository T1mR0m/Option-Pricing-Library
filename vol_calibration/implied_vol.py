import numpy as np

class ImpliedVolCalibrator():
    """
    Expects an instantiated Option object (e.g., EuropeanOption) 
    and the observed market price of that option.
    """
    def __init__(self, option, target_price):
        self.option = option
        self.target_price = target_price

    def calc_iv_newton_raphson(self, initial_vol=0.2, tol=1e-5, max_iter=100):
        """
        Calculates implied volatility using the Newton-Raphson method.
        """
        sigma = initial_vol

        for i in range(max_iter):
            self.option.sigma = sigma
            curr_price = self.option.price()

            if abs(curr_price - self.target_price) <= tol:
                print(f'\nNewton-Raphson estimator converged after {i + 1} iterations')
                return float(sigma)

            vega = self.option.vega() * 100

            if vega > 1e-3:
                sigma = sigma - ((curr_price - self.target_price) / vega)

                sigma = float(np.clip(sigma, 1e-4, 5.0))
                    
            else:
                raise ValueError('Vega converged to extremelly small value')

        raise ValueError('Newton-Raphson estimator failed to converge')