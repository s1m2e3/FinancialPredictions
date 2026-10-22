# Growth-pool signal audit (fair sample)

Do any of the cheap features pick which NEW companies will go on to grow? A signal audit
inside the growth pool's fair sample (growth_pool.py --sample: companies that first reached
$2B in 2013 or later, listed and gone alike).

POINT IN TIME, both ways:
    membership   a company is in the pool on day t only if its market value that day was at
                 least FLOOR: shares outstanding AS FILED with the SEC before t (EDGAR,
                 dei:EntityCommonStockSharesOutstanding, all classes) x the price it was
                 quoted at the day before -- never the size it reached later
    features     computed from prices up to the day before (trend_features.py and below)
    outcomes     open-to-open returns over the next h days; a stock that stops trading
                 inside the window keeps its last price (a buy-out pays about the deal price,
                 a failure about nothing), so the dead are never dropped from the outcome

Prices: yfinance for the listed (data/yf, yf_prices.py), Tiingo for the gone (data/tiingo,
tiingo_prices.py), adjusted for splits and dividends; Tiingo's stale rows after a delisting
(no volume) are cut. The rank IC, its Newey-West t and the Bonferroni bar are signal_audit's.

```
horizon                        input  IC_1419  t_1419  days_1419  IC_2026  t_2026  days_2026
    21d                      ma_5_21   -0.027  -1.128         71    0.019   1.322         80
    21d                     ma_21_63    0.009   0.428         71    0.009   0.475         80
    21d                    ma_50_200    0.025   1.148         71    0.049   2.272         80
    21d                     px_ma_50   -0.022  -0.962         71    0.011   0.654         80
    21d                    px_ma_200    0.021   1.010         71    0.049   2.311         80
    21d                  ma_slope_21   -0.012  -0.582         71    0.021   1.212         80
    21d                  ma_slope_63    0.002   0.108         71    0.017   0.856         80
    21d        slope_acc_63 (2 diff)    0.012   0.532         71   -0.000  -0.014         80
    21d                near_52w_high    0.047   2.066         71    0.037   1.813         80
    21d                above_52w_low    0.010   0.437         71    0.057   2.730         80
    21d             dd_from_63d_high   -0.006  -0.249         71    0.022   1.237         80
    21d       overhang (P / 1y VWAP)    0.032   1.586         71    0.053   2.532         80
    21d                     mom_12_1    0.024   0.984         71    0.033   1.722         80
    21d                       rev_1m    0.028   1.283         71   -0.019  -1.219         80
    21d       gain_since_first_trade    0.015   0.746         71    0.024   1.619         80
    21d      vol_21 (close to close)   -0.022  -1.052         71   -0.007  -0.336         80
    21d vol_21 (range, Garman-Klass)   -0.023  -1.015         71   -0.005  -0.215         80
    21d        log_age (days traded)   -0.009  -0.489         71    0.041   2.626         80
    21d             log_market_value   -0.047  -2.007         71   -0.030  -1.702         80
    21d                 mv_growth_1y    0.002   0.087         71    0.033   1.823         80
    63d                      ma_5_21   -0.011  -0.451         69    0.015   0.877         78
    63d                     ma_21_63    0.000   0.004         69    0.026   1.111         78
    63d                    ma_50_200    0.059   2.119         69    0.085   3.675         78
    63d                     px_ma_50   -0.012  -0.485         69    0.016   0.731         78
    63d                    px_ma_200    0.043   1.654         69    0.079   3.192         78
    63d                  ma_slope_21   -0.010  -0.471         69    0.014   0.753         78
    63d                  ma_slope_63    0.014   0.591         69    0.033   1.348         78
    63d        slope_acc_63 (2 diff)   -0.030  -1.085         69   -0.016  -0.933         78
    63d                near_52w_high    0.095   3.769         69    0.056   2.327         78
    63d                above_52w_low    0.037   1.371         69    0.087   2.890         78
    63d             dd_from_63d_high    0.029   1.308         69    0.038   1.736         78
    63d       overhang (P / 1y VWAP)    0.062   2.436         69    0.087   3.528         78
    63d                     mom_12_1    0.040   1.491         69    0.055   1.713         78
    63d                       rev_1m    0.019   0.853         69   -0.016  -0.836         78
    63d       gain_since_first_trade    0.018   0.628         69    0.040   1.489         78
    63d      vol_21 (close to close)   -0.051  -1.777         69   -0.021  -0.641         78
    63d vol_21 (range, Garman-Klass)   -0.065  -2.024         69   -0.023  -0.578         78
    63d        log_age (days traded)   -0.030  -1.496         69    0.069   2.305         78
    63d             log_market_value   -0.078  -2.394         69   -0.037  -1.250         78
    63d                 mv_growth_1y    0.013   0.558         69    0.045   1.644         78
   252d                      ma_5_21   -0.004  -0.171         60    0.033   2.598         69
   252d                     ma_21_63   -0.000  -0.014         60    0.061   3.135         69
   252d                    ma_50_200    0.001   0.041         60    0.087   1.884         69
   252d                     px_ma_50    0.011   0.426         60    0.055   2.802         69
   252d                    px_ma_200    0.003   0.095         60    0.092   2.193         69
   252d                  ma_slope_21   -0.005  -0.319         60    0.046   3.069         69
   252d                  ma_slope_63   -0.008  -0.301         60    0.081   3.039         69
   252d        slope_acc_63 (2 diff)    0.001   0.064         60    0.007   0.580         69
   252d                near_52w_high    0.075   2.888         60    0.088   1.609         69
   252d                above_52w_low    0.017   0.630         60    0.106   4.775         69
   252d             dd_from_63d_high    0.030   0.919         60    0.068   1.587         69
   252d       overhang (P / 1y VWAP)    0.046   1.924         60    0.099   2.323         69
   252d                     mom_12_1    0.052   2.193         60    0.059   1.002         69
   252d                       rev_1m    0.015   0.921         60   -0.045  -2.995         69
   252d       gain_since_first_trade    0.022   0.355         60    0.049   0.893         69
   252d      vol_21 (close to close)   -0.049  -0.953         60   -0.023  -0.368         69
   252d vol_21 (range, Garman-Klass)   -0.073  -1.203         60   -0.045  -0.567         69
   252d        log_age (days traded)   -0.048  -3.093         60    0.165   3.767         69
   252d             log_market_value   -0.167  -3.624         60   -0.089  -1.507         69
   252d                 mv_growth_1y   -0.013  -0.362         60    0.049   1.031         69
```

Bonferroni bar |t| > 3.34

## Portfolio check (equal weights, 21-day rebalance, 5 bp costs)

```
   period                     portfolio  return %/yr  vol %  max DD %  beta  alpha %/yr  t(alpha)  periods
2014-2019    top third by near_52w_high        10.75  14.51    -16.38  0.93        0.14      0.04       72
2014-2019                    whole pool         9.22  16.83    -22.48  1.15       -3.61     -0.97       72
2014-2019 bottom third by near_52w_high         6.81  22.52    -29.51  1.39       -7.61     -1.26       72
2014-2019                 S&P 500 (SPY)        11.87  12.31    -14.93  1.00        0.00      2.25       72
2014-2019         top third by mom_12_1        10.01  19.54    -30.60  1.09       -1.59     -0.27       72
2014-2019      bottom third by mom_12_1         8.21  21.24    -31.23  1.27       -5.10     -0.86       72
2020-2026    top third by near_52w_high        14.65  21.45    -28.09  1.00        0.16      0.04       80
2020-2026                    whole pool        11.05  24.15    -36.56  1.19       -5.36     -1.27       80
2020-2026 bottom third by near_52w_high         4.22  31.58    -44.76  1.43      -13.79     -1.99       80
2020-2026                 S&P 500 (SPY)        15.21  18.23    -23.77  1.00        0.00      2.27       80
2020-2026         top third by mom_12_1        15.68  23.17    -28.68  1.07        0.38      0.08       80
2020-2026      bottom third by mom_12_1         5.94  29.21    -42.34  1.32      -10.93     -1.68       80
```
