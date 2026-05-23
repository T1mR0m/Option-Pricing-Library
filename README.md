# Option-Pricing-Library
A Python library for pricing and risk analysis of a wide range of options built from scratch.

An object-oriented quantitative finance library for pricing vanilla and exotic options, complete with risk sensitivities (Greeks) and live market data integration. 

This project implements standard industry pricing models from scratch, avoiding black-box libraries, to demonstrate the mathematical and computational mechanics of derivative pricing.

## Features

* **Supported Option Types:**
  * **European:** Black-Scholes analytical & Monte Carlo.
  * **American:** Cox-Ross-Rubinstein (CRR) Binomial Tree.
  * **Barrier (Knock-in/out):** Monte Carlo with Broadie-Glasserman-Kou (BGK) continuity correction.
  * **Asian (Arithmetic):** Path-dependent Monte Carlo.
  * **Binary (Cash-or-nothing):** Analytical & Monte Carlo.
  * **Autocallables:** Structured product pricing with coupon payouts and downside protection.
* **Risk Sensitivities (Greeks):**
  * Exact analytical Greeks for Black-Scholes.
  * Finite Difference Methods (Forward and Central bump & reprice) for exotics and trees.
* **Computational Optimization:**
  * Fully vectorized `numpy` Monte Carlo simulations.
  * Antithetic variates for variance reduction in stochastic paths.
* **Live Market Integration:**
  * Automatic parameter fetching (Underlying Price, Implied Volatility, Risk-Free Rate, Dividend Yield) via `yfinance`.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/T1mR0m/Option-Pricing-Library.git](https://github.com/T1mR0m/Option-Pricing-Library.git)
   cd Option-Pricing-Library
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Demo Usage
The library features a clean OOP hierarchy. You can price a custom exotic option manually, or fetch live data directly from the market.

```python
from option_pricing import EuropeanOption, BarrierOption, build_option_from_ticker

# 1. LIVE MARKET DATA (European Call on AAPL)
aapl_call = build_option_from_ticker(
    ticker="AAPL", 
    K=185.0, 
    expiration_date="2026-06-19", 
    opt_type="call", 
    option_class=EuropeanOption
)
print(f"BS Price: ${aapl_call.price_analytical():.2f}")
print(f"Delta: {aapl_call.greeks()['delta']:.4f}")

# 2. PRICING AN EXOTIC (Down-and-Out Put)
barrier_put = BarrierOption(
    S=100.0, K=100.0, B=90.0, 
    sigma=0.20, r=0.05, q=0.0, T=1.0, 
    opt_type="put", barrier_type="down-and-out"
)
mc_price, mc_se = barrier_put.price()
print(f"MC Price: ${mc_price:.4f} (SE: {mc_se:.4f})")
```

## Architecture

All derivative products inherit from a base `Option` class. This unified interface allows you to easily compute `.price()` and `.greeks()` regardless of the underlying mathematical complexity (Analytical vs. Tree vs. Simulation). 

For a deep dive into the mathematics, stochastic calculus, and discretization logic behind the code, please refer to the extensively documented `option_pricing.ipynb` notebook included in this repository.

## Disclaimer
This library is for educational and academic purposes only. Do not use this code to make actual financial or trading decisions.
