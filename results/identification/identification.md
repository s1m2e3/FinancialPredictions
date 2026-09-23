# SARIMAX identification (Box-Jenkins), training window 2013-01-04 .. 2020-12-31, 2013 days

## 1. Regular differencing

| series | ADF p (H0 unit root) | Phillips-Perron p (H0 unit root) | KPSS p (H0 stationary) |
|---|---|---|---|
| level: 100 log P | 0.2335 | 0.1693 | 0.0100 |
| first difference | 0.0000 | 0.0000 | 0.1000 |

pmdarima `ndiffs`: KPSS 1, ADF 1, PP 1. KPSS p-values are truncated to [0.01, 0.10] by the published tables. **d = 1**.

## 2-3. Seasonality

| season | OCSB D | Canova-Hansen D | STL strength, log price | STL strength, differenced | Ljung-Box s,2s,3s p | robust Ljung-Box p |
|---|---|---|---|---|---|---|
| 5 (week) | 0.0000 | 0.0000 | 0.0000 | 0.0915 | 0.0000 | 0.0653 |
| 21 (month) | 0.0000 | 0.0000 | 0.0276 | 0.1452 | 0.0181 | 0.1034 |
| 63 (quarter) | 0.0000 | 0.0000 | 0.0000 | 0.1188 | 0.4914 | 0.6467 |
| 252 (year) | 0.0000 | 0.0000 | 0.1046 | 0.1383 | 0.9270 | 0.8746 |

STL strength above 0.64 is Hyndman's threshold for seasonal differencing. The robust Ljung-Box uses heteroskedasticity-robust autocorrelation variances; the classic version over-rejects when volatility clusters.

Periodogram of the differenced series: largest peak at a period of 2.3 days, Fisher's g = 0.0089, p = 0.121 (H0: no periodic component).

Seasonal differencing used (OCSB): s=5: D=0, s=21: D=0, s=63: D=0.

## 4. Regressors (one OLS with all blocks, Newey-West HAC, 10 lags)

| block | regressors | HAC Wald p | included |
|---|---|---|---|
| day of week | 4 | 0.5497 | False |
| month of year | 11 | 0.6266 | False |
| turn of month | 1 | 0.9273 | False |
| lagged constituents | 9 | 0.2135 | False |

Regressors carried into the model: none.

## 5. Order search

900 Gaussian SARIMA models, common sample of 1881 days (2013-07-16 on). Ten best by AICc:

| model | params | NLL / day | AIC | AICc | BIC | dAICc | dBIC |
|---|---|---|---|---|---|---|---|
| SARIMA(4,1,4)(2,0,2)[5] | 14.00 | 1.58 | 5979.90 | 5980.13 | 6057.46 | 0.00 | 33.36 |
| SARIMA(5,1,5)(1,0,1)[5] | 14.00 | 1.58 | 5983.74 | 5983.96 | 6061.29 | 3.84 | 37.20 |
| SARIMA(2,1,2)(1,0,1)[5] | 8.00 | 1.59 | 5984.15 | 5984.23 | 6028.47 | 4.10 | 4.37 |
| SARIMA(2,1,2)(2,0,1)[5] | 9.00 | 1.59 | 5984.81 | 5984.90 | 6034.66 | 4.77 | 10.56 |
| SARIMA(3,1,3)(1,0,0)[5] | 9.00 | 1.59 | 5984.91 | 5985.01 | 6034.77 | 4.88 | 10.67 |
| SARIMA(3,1,3)(0,0,1)[5] | 9.00 | 1.59 | 5985.02 | 5985.11 | 6034.87 | 4.98 | 10.77 |
| SARIMA(2,1,2)(1,0,2)[5] | 9.00 | 1.59 | 5985.03 | 5985.13 | 6034.89 | 5.00 | 10.79 |
| SARIMA(2,1,2)(0,0,1)[5] | 7.00 | 1.59 | 5985.32 | 5985.38 | 6024.10 | 5.25 | 0.00 |
| SARIMA(2,1,2)(2,0,2)[5] | 10.00 | 1.59 | 5985.88 | 5986.00 | 6041.27 | 5.87 | 17.18 |
| SARIMA(3,1,2)(1,0,1)[5] | 9.00 | 1.59 | 5986.08 | 5986.18 | 6035.94 | 6.05 | 11.84 |

Ten best by BIC:

| model | params | NLL / day | AIC | AICc | BIC | dAICc | dBIC |
|---|---|---|---|---|---|---|---|
| SARIMA(2,1,2)(0,0,1)[5] | 7.00 | 1.59 | 5985.32 | 5985.38 | 6024.10 | 5.25 | 0.00 |
| SARIMA(2,1,2) | 6.00 | 1.59 | 5991.60 | 5991.64 | 6024.84 | 11.51 | 0.74 |
| SARIMA(2,1,2)(1,0,0)[5] | 7.00 | 1.59 | 5986.34 | 5986.39 | 6025.11 | 6.27 | 1.02 |
| SARIMA(2,1,2)(1,0,1)[5] | 8.00 | 1.59 | 5984.15 | 5984.23 | 6028.47 | 4.10 | 4.37 |
| SARIMA(3,1,2)(0,0,1)[5] | 8.00 | 1.59 | 5986.38 | 5986.46 | 6030.70 | 6.33 | 6.60 |
| SARIMA(2,1,2)(0,0,1)[21] | 7.00 | 1.59 | 5992.13 | 5992.18 | 6030.90 | 12.06 | 6.81 |
| SARIMA(2,1,2)(1,0,0)[21] | 7.00 | 1.59 | 5992.14 | 5992.20 | 6030.92 | 12.07 | 6.82 |
| SARIMA(2,1,3)(0,0,1)[5] | 8.00 | 1.59 | 5987.09 | 5987.16 | 6031.40 | 7.03 | 7.31 |
| SARIMA(3,1,2)(1,0,0)[5] | 8.00 | 1.59 | 5987.22 | 5987.30 | 6031.54 | 7.17 | 7.44 |
| SARIMA(2,1,2)(0,0,1)[63] | 7.00 | 1.59 | 5992.79 | 5992.85 | 6031.57 | 12.72 | 7.47 |

**Selected by AICc: SARIMA(4,1,4)(2,0,2)[5]**. Best by BIC: SARIMA(2,1,2)(0,0,1)[5]. Best non-seasonal by AICc: SARIMA(3,1,3) (dAICc 11.24).

## 6. Diagnostics of the selected model (training residuals)

| test | p-value |
|---|---|
| Ljung-Box residuals, 21 lags | 0.11 |
| Ljung-Box residuals, 42 lags | 0.182 |
| Ljung-Box residuals, 63 lags | 0.0721 |
| robust Ljung-Box residuals, 21 lags | 0.736 |
| robust Ljung-Box residuals, 42 lags | 0.874 |
| robust Ljung-Box residuals, 63 lags | 0.624 |
| robust Ljung-Box s=5 lags s,2s,3s | 0.981 |
| robust Ljung-Box s=21 lags s,2s,3s | 0.549 |
| robust Ljung-Box s=63 lags s,2s,3s | 0.722 |
| Ljung-Box squared residuals, 10 lags (ARCH) | 4e-304 |
| ARCH-LM, 5 lags | 1.58e-116 |
| Jarque-Bera (normality) | 0 |

Residual skewness -1.24, excess kurtosis 9.76.

Residual autocorrelation (robust Ljung-Box, valid under ARCH): none left at 5%. ARCH effects in the residuals: yes; normality rejected.

Fitted coefficients: c=0.0706, phi=[-0.0898, 1.0028, -0.2097, -0.8386], Phi=[0.4408, 0.2542], theta=[-0.0374, -0.9626, 0.3459, 0.7162], Theta=[-0.5378, -0.2109]
