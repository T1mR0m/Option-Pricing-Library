# Option-Pricing-Library
A Python library for pricing and risk analysis of a wide range of options built from scratch.

# Python Option Pricing & Risk Library

An object-oriented quantitative finance library for pricing vanilla and exotic options, complete with risk sensitivities (Greeks) and live market data integration. 

This project implements standard industry pricing models from scratch, avoiding black-box libraries, to demonstrate the mathematical and computational mechanics of derivative pricing.

## 🚀 Features

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

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/T1mR0m/Option-Pricing-Library.git](https://github.com/T1mR0m/Option-Pricing-Library.git)
   cd option-pricing-lib
