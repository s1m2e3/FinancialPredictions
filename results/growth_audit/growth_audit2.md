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
    21d                      ma_5_21   -0.024  -1.062         71    0.009   0.526         80
    21d                     ma_21_63   -0.013  -0.500         71    0.007   0.370         80
    21d                    ma_50_200   -0.019  -0.762         71    0.016   0.786         80
    21d                     px_ma_50   -0.018  -0.751         71    0.001   0.050         80
    21d                    px_ma_200   -0.021  -0.844         71    0.019   0.883         80
    21d                  ma_slope_21   -0.007  -0.342         71    0.019   1.131         80
    21d                  ma_slope_63   -0.025  -0.857         71   -0.004  -0.173         80
    21d        slope_acc_63 (2 diff)    0.030   1.292         71    0.003   0.191         80
    21d                near_52w_high   -0.007  -0.291         71    0.025   1.285         80
    21d                above_52w_low   -0.002  -0.078         71    0.001   0.064         80
    21d             dd_from_63d_high   -0.000  -0.007         71    0.004   0.188         80
    21d       overhang (P / 1y VWAP)   -0.009  -0.345         71    0.024   1.106         80
    21d                     mom_12_1    0.009   0.364         71    0.016   0.820         80
    21d                       rev_1m    0.020   0.889         71   -0.017  -1.030         80
    21d       gain_since_first_trade   -0.007  -0.335         71    0.048   2.782         80
    21d      vol_21 (close to close)   -0.018  -0.736         71   -0.023  -0.920         80
    21d vol_21 (range, Garman-Klass)   -0.028  -1.048         71   -0.029  -1.090         80
    21d        log_age (days traded)   -0.030  -1.596         71    0.018   1.014         80
    21d             log_market_value   -0.013  -0.594         71   -0.005  -0.316         80
    21d                 mv_growth_1y   -0.006  -0.242         71    0.016   0.767         80
    63d                      ma_5_21    0.003   0.115         69    0.002   0.102         78
    63d                     ma_21_63   -0.029  -0.788         69    0.003   0.128         78
    63d                    ma_50_200   -0.007  -0.214         69    0.039   1.604         78
    63d                     px_ma_50   -0.017  -0.473         69   -0.004  -0.144         78
    63d                    px_ma_200   -0.007  -0.171         69    0.028   1.036         78
    63d                  ma_slope_21    0.001   0.057         69    0.001   0.057         78
    63d                  ma_slope_63   -0.040  -1.070         69    0.009   0.310         78
    63d        slope_acc_63 (2 diff)   -0.007  -0.225         69   -0.009  -0.597         78
    63d                near_52w_high    0.019   0.648         69    0.038   1.183         78
    63d                above_52w_low   -0.004  -0.112         69    0.005   0.171         78
    63d             dd_from_63d_high    0.018   0.560         69    0.004   0.147         78
    63d       overhang (P / 1y VWAP)    0.004   0.098         69    0.032   1.090         78
    63d                     mom_12_1   -0.015  -0.475         69    0.017   0.604         78
    63d                       rev_1m    0.000   0.003         69   -0.002  -0.084         78
    63d       gain_since_first_trade   -0.007  -0.211         69    0.072   2.460         78
    63d      vol_21 (close to close)   -0.030  -0.805         69   -0.037  -0.928         78
    63d vol_21 (range, Garman-Klass)   -0.028  -0.740         69   -0.046  -1.220         78
    63d        log_age (days traded)   -0.041  -1.550         69    0.040   1.366         78
    63d             log_market_value   -0.025  -0.661         69   -0.009  -0.410         78
    63d                 mv_growth_1y   -0.016  -0.451         69    0.011   0.408         78
   252d                      ma_5_21    0.028   1.267         60    0.010   0.437         69
   252d                     ma_21_63    0.015   0.428         60    0.027   1.115         69
   252d                    ma_50_200    0.031   0.589         60    0.047   1.135         69
   252d                     px_ma_50    0.023   0.725         60    0.021   0.832         69
   252d                    px_ma_200    0.037   0.673         60    0.043   0.964         69
   252d                  ma_slope_21    0.022   0.967         60    0.024   1.297         69
   252d                  ma_slope_63    0.004   0.081         60    0.034   1.218         69
   252d        slope_acc_63 (2 diff)   -0.005  -0.233         60   -0.005  -0.387         69
   252d                near_52w_high    0.072   1.657         60    0.091   1.471         69
   252d                above_52w_low   -0.006  -0.099         60    0.029   0.875         69
   252d             dd_from_63d_high    0.064   1.768         60    0.045   0.793         69
   252d       overhang (P / 1y VWAP)    0.054   0.953         60    0.049   1.032         69
   252d                     mom_12_1   -0.002  -0.029         60    0.033   0.579         69
   252d                       rev_1m   -0.016  -0.679         60   -0.019  -1.060         69
   252d       gain_since_first_trade   -0.023  -0.370         60    0.098   2.088         69
   252d      vol_21 (close to close)   -0.049  -1.183         60   -0.055  -0.789         69
   252d vol_21 (range, Garman-Klass)   -0.038  -0.795         60   -0.083  -1.214         69
   252d        log_age (days traded)   -0.058  -1.210         60    0.068   1.236         69
   252d             log_market_value   -0.063  -0.884         60   -0.002  -0.075         69
   252d                 mv_growth_1y   -0.026  -0.371         60    0.026   0.492         69
```

Bonferroni bar |t| > 3.34

## Portfolio check (equal weights, 21-day rebalance, 5 bp costs)

```
   period                     portfolio  return %/yr  vol %  max DD %  beta  alpha %/yr  t(alpha)  periods
2014-2019    top third by near_52w_high         8.11  15.16    -22.99  1.03       -3.44     -1.01       72
2014-2019                    whole pool        11.40  16.03    -19.71  1.15       -1.73     -0.57       72
2014-2019 bottom third by near_52w_high        11.70  21.28    -27.58  1.42       -3.77     -0.76       72
2014-2019                 S&P 500 (SPY)        11.87  12.31    -14.93  1.00        0.00      2.25       72
2014-2019         top third by mom_12_1        12.36  17.20    -23.90  1.12       -0.27     -0.07       72
2014-2019      bottom third by mom_12_1        12.05  19.27    -23.66  1.31       -2.52     -0.58       72
2020-2026    top third by near_52w_high        11.55  19.39    -27.51  0.89       -1.34     -0.33       80
2020-2026                    whole pool        10.33  21.53    -30.11  1.07       -4.85     -1.38       80
2020-2026 bottom third by near_52w_high         2.74  27.13    -44.56  1.27      -13.92     -2.54       80
2020-2026                 S&P 500 (SPY)        15.21  18.23    -23.77  1.00        0.00      2.27       80
2020-2026         top third by mom_12_1        15.18  23.76    -24.93  1.08       -0.11     -0.02       80
2020-2026      bottom third by mom_12_1         5.48  26.07    -36.70  1.21      -10.43     -1.91       80
```
