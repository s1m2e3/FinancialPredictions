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
horizon                                     input  IC_1419  t_1419  days_1419  IC_2026  t_2026  days_2026
    21d                                   ma_5_21   -0.027  -1.333         71    0.015   1.142         80
    21d                                  ma_21_63   -0.005  -0.322         71    0.009   0.534         80
    21d                                 ma_50_200    0.005   0.234         71    0.034   1.805         80
    21d                                  px_ma_50   -0.019  -0.994         71    0.006   0.467         80
    21d                                 px_ma_200    0.001   0.067         71    0.036   1.963         80
    21d                               ma_slope_21   -0.014  -0.769         71    0.022   1.500         80
    21d                               ma_slope_63   -0.009  -0.523         71    0.008   0.450         80
    21d                     slope_acc_63 (2 diff)    0.019   1.019         71    0.000   0.008         80
    21d                             near_52w_high    0.022   1.095         71    0.032   1.693         80
    21d                             above_52w_low    0.002   0.106         71    0.029   1.521         80
    21d                          dd_from_63d_high   -0.003  -0.165         71    0.014   0.827         80
    21d                    overhang (P / 1y VWAP)    0.009   0.451         71    0.041   2.132         80
    21d                                  mom_12_1    0.016   0.779         71    0.025   1.410         80
    21d                                    rev_1m    0.027   1.437         71   -0.020  -1.442         80
    21d                    gain_since_first_trade    0.005   0.296         71    0.035   2.537         80
    21d                   vol_21 (close to close)   -0.021  -1.139         71   -0.016  -0.701         80
    21d              vol_21 (range, Garman-Klass)   -0.027  -1.306         71   -0.017  -0.714         80
    21d                     log_age (days traded)   -0.016  -1.272         71    0.030   1.987         80
    21d                          log_market_value   -0.028  -1.609         71   -0.018  -1.366         80
    21d                              mv_growth_1y   -0.004  -0.187         71    0.026   1.438         80
    21d                   abn_volume (21d / 252d)    0.011   0.990         71   -0.008  -0.757         80
    21d                     abn_volume (5d / 63d)    0.006   0.498         71   -0.004  -0.308         80
    21d                 volume_trend (63d / 252d)    0.005   0.446         71   -0.010  -1.022         80
    21d                 turnover (63d, of shares)   -0.019  -1.043         71   -0.006  -0.311         80
    21d                   log_dollar_volume (63d)   -0.023  -1.560         71   -0.013  -1.118         80
    21d                     rev_growth (TTM, y/y)   -0.011  -0.575         71    0.016   0.882         80
    21d rev_accel (growth - growth a quarter ago)    0.000   0.023         71   -0.003  -0.233         80
    21d                        gross_margin (TTM)   -0.006  -0.270         71   -0.032  -1.424         80
    21d                           op_margin (TTM)    0.021   0.976         71    0.015   0.746         80
    21d                       profit_margin (TTM)    0.016   0.857         71    0.017   0.953         80
    21d                        asset_growth (y/y)   -0.009  -0.596         71    0.005   0.275         80
    21d                   sue (earnings surprise)   -0.007  -0.570         71    0.015   1.287         80
    21d                rev_sue (revenue surprise)    0.049   3.188         71    0.010   0.554         80
    63d                                   ma_5_21   -0.002  -0.126         69    0.009   0.632         78
    63d                                  ma_21_63   -0.017  -0.626         69    0.017   0.767         78
    63d                                 ma_50_200    0.029   1.068         69    0.066   3.156         78
    63d                                  px_ma_50   -0.013  -0.520         69    0.006   0.315         78
    63d                                 px_ma_200    0.020   0.636         69    0.057   2.554         78
    63d                               ma_slope_21   -0.006  -0.313         69    0.010   0.595         78
    63d                               ma_slope_63   -0.011  -0.419         69    0.024   1.022         78
    63d                     slope_acc_63 (2 diff)   -0.017  -0.718         69   -0.015  -1.157         78
    63d                             near_52w_high    0.061   2.855         69    0.049   1.865         78
    63d                             above_52w_low    0.014   0.455         69    0.049   1.868         78
    63d                          dd_from_63d_high    0.024   1.022         69    0.023   1.026         78
    63d                    overhang (P / 1y VWAP)    0.030   1.004         69    0.063   2.677         78
    63d                                  mom_12_1    0.014   0.534         69    0.038   1.377         78
    63d                                    rev_1m    0.011   0.592         69   -0.011  -0.689         78
    63d                    gain_since_first_trade    0.011   0.418         69    0.054   2.274         78
    63d                   vol_21 (close to close)   -0.045  -1.646         69   -0.029  -0.823         78
    63d              vol_21 (range, Garman-Klass)   -0.052  -1.808         69   -0.033  -0.865         78
    63d                     log_age (days traded)   -0.030  -1.916         69    0.055   1.966         78
    63d                          log_market_value   -0.047  -1.795         69   -0.023  -1.123         78
    63d                              mv_growth_1y   -0.005  -0.206         69    0.032   1.308         78
    63d                   abn_volume (21d / 252d)    0.011   0.762         69   -0.016  -1.483         78
    63d                     abn_volume (5d / 63d)   -0.006  -0.425         69   -0.002  -0.193         78
    63d                 volume_trend (63d / 252d)    0.003   0.177         69   -0.020  -1.680         78
    63d                 turnover (63d, of shares)   -0.042  -1.467         69   -0.014  -0.522         78
    63d                   log_dollar_volume (63d)   -0.049  -2.288         69   -0.025  -1.191         78
    63d                     rev_growth (TTM, y/y)   -0.032  -1.256         69    0.002   0.063         78
    63d rev_accel (growth - growth a quarter ago)   -0.003  -0.107         69   -0.004  -0.234         78
    63d                        gross_margin (TTM)    0.002   0.057         69   -0.074  -2.522         78
    63d                           op_margin (TTM)    0.036   1.131         69    0.024   0.717         78
    63d                       profit_margin (TTM)    0.035   1.137         69    0.022   0.719         78
    63d                        asset_growth (y/y)   -0.019  -0.812         69    0.003   0.107         78
    63d                   sue (earnings surprise)    0.003   0.185         69    0.027   1.447         78
    63d                rev_sue (revenue surprise)    0.075   3.421         69    0.000   0.002         78
   252d                                   ma_5_21    0.010   0.507         60    0.023   1.497         69
   252d                                  ma_21_63    0.002   0.080         60    0.046   2.301         69
   252d                                 ma_50_200    0.016   0.468         60    0.069   1.647         69
   252d                                  px_ma_50    0.016   0.619         60    0.040   1.948         69
   252d                                 px_ma_200    0.022   0.565         60    0.070   1.692         69
   252d                               ma_slope_21    0.006   0.343         60    0.037   2.644         69
   252d                               ma_slope_63   -0.004  -0.108         60    0.059   2.391         69
   252d                     slope_acc_63 (2 diff)   -0.001  -0.095         60    0.002   0.184         69
   252d                             near_52w_high    0.076   2.645         60    0.092   1.582         69
   252d                             above_52w_low    0.003   0.072         60    0.072   3.370         69
   252d                          dd_from_63d_high    0.043   1.338         60    0.059   1.179         69
   252d                    overhang (P / 1y VWAP)    0.049   1.315         60    0.077   1.831         69
   252d                                  mom_12_1    0.027   0.689         60    0.047   0.878         69
   252d                                    rev_1m    0.002   0.110         60   -0.034  -2.424         69
   252d                    gain_since_first_trade    0.008   0.146         60    0.069   1.592         69
   252d                   vol_21 (close to close)   -0.052  -1.217         60   -0.040  -0.610         69
   252d              vol_21 (range, Garman-Klass)   -0.058  -1.159         60   -0.066  -0.903         69
   252d                     log_age (days traded)   -0.039  -1.617         60    0.121   2.417         69
   252d                          log_market_value   -0.112  -2.235         60   -0.048  -1.439         69
   252d                              mv_growth_1y   -0.015  -0.309         60    0.041   0.868         69
   252d                   abn_volume (21d / 252d)   -0.024  -1.647         60    0.005   0.328         69
   252d                     abn_volume (5d / 63d)    0.004   0.345         60    0.014   1.027         69
   252d                 volume_trend (63d / 252d)   -0.034  -1.947         60   -0.003  -0.211         69
   252d                 turnover (63d, of shares)   -0.058  -1.386         60   -0.040  -0.830         69
   252d                   log_dollar_volume (63d)   -0.092  -2.003         60   -0.052  -2.066         69
   252d                     rev_growth (TTM, y/y)   -0.077  -1.284         60   -0.028  -0.465         69
   252d rev_accel (growth - growth a quarter ago)   -0.030  -0.853         60    0.015   0.930         69
   252d                        gross_margin (TTM)   -0.037  -0.851         60   -0.188  -5.273         69
   252d                           op_margin (TTM)    0.017   0.558         60    0.099   1.927         69
   252d                       profit_margin (TTM)    0.048   1.277         60    0.078   1.662         69
   252d                        asset_growth (y/y)   -0.045  -0.663         60   -0.004  -0.084         69
   252d                   sue (earnings surprise)    0.053   1.864         60    0.063   1.802         69
   252d                rev_sue (revenue surprise)    0.116   2.106         60   -0.036  -0.731         69
```

Bonferroni bar |t| > 3.48

## Portfolio check (21-day decisions, 5 bp costs; the thirds equal-weighted and rebalanced; RIDE never trims a winner and sells only 30% below the 52-week high)

```
   period                                                         portfolio  return %/yr  vol %  max DD %  beta  alpha %/yr  t(alpha)  periods
2014-2019                                        top third by near_52w_high         9.45  14.23    -20.28  0.99       -1.78     -0.58       72
2014-2019                                                        whole pool        10.41  16.12    -21.07  1.15       -2.62     -0.83       72
2014-2019                                     bottom third by near_52w_high         7.79  21.19    -26.55  1.42       -7.33     -1.49       72
2014-2019                                                     S&P 500 (SPY)        11.87  12.31    -14.93  1.00        0.00      2.25       72
2014-2019         RIDE: enter top third by near_52w_high, exit 30% off high        10.14  13.58    -17.73  0.96       -0.90     -0.33       72
2014-2019                                             top third by mom_12_1        11.44  17.63    -26.08  1.12       -1.02     -0.23       72
2014-2019                                          bottom third by mom_12_1        10.14  19.47    -24.78  1.31       -4.21     -0.94       72
2014-2019              RIDE: enter top third by mom_12_1, exit 30% off high        10.37  14.96    -19.17  1.02       -1.20     -0.36       72
2014-2019                                top third by rev_growth (TTM, y/y)         8.17  19.64    -25.02  1.29       -5.76     -1.22       72
2014-2019                             bottom third by rev_growth (TTM, y/y)        10.97  17.61    -22.81  1.19       -2.30     -0.57       72
2014-2019 RIDE: enter top third by rev_growth (TTM, y/y), exit 30% off high         7.67  17.01    -20.70  1.16       -5.10     -1.35       72
2020-2026                                        top third by near_52w_high        13.23  19.71    -28.47  0.94       -0.51     -0.14       80
2020-2026                                                        whole pool        10.77  22.59    -33.33  1.13       -5.08     -1.39       80
2020-2026                                     bottom third by near_52w_high         2.41  28.42    -42.05  1.35      -15.02     -2.69       80
2020-2026                                                     S&P 500 (SPY)        15.21  18.23    -23.77  1.00        0.00      2.27       80
2020-2026         RIDE: enter top third by near_52w_high, exit 30% off high        11.65  19.33    -28.61  0.93       -1.78     -0.49       80
2020-2026                                             top third by mom_12_1        15.70  22.73    -27.53  1.07        0.27      0.06       80
2020-2026                                          bottom third by mom_12_1         5.57  27.02    -39.27  1.26      -11.03     -2.01       80
2020-2026              RIDE: enter top third by mom_12_1, exit 30% off high        12.89  20.35    -24.76  0.97       -1.18     -0.30       80
2020-2026                                top third by rev_growth (TTM, y/y)        12.99  24.66    -37.64  1.16       -3.16     -0.64       80
2020-2026                             bottom third by rev_growth (TTM, y/y)         7.14  23.70    -41.40  1.11       -7.83     -1.63       80
2020-2026 RIDE: enter top third by rev_growth (TTM, y/y), exit 30% off high        13.46  22.85    -32.74  1.09       -2.10     -0.48       80
```
