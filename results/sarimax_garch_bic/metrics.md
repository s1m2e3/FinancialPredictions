# SARIMAX + GARCH, 1 day to 1 month ahead: results

Mean equation from identification.py, lowest BIC: **SARIMA(2,1,2)(0,0,1)[5]**.

| window | dates | days |
|---|---|---|
| train | 2013-01-04 .. 2020-12-31 | 2013 |
| validation | .. 2022-12-31 | 503 |
| test | .. 2025-12-31 | 752 (walk-forward, refit every 63 days, 5000 simulated paths per origin) |

## Parameters (train fit)

| parameter (train fit) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| c | 0.0658 | 0.1374 | 0.0658 | 0.1117 | 0.0706 | -0.0814 |
| phi1 | -1.7296 | -1.6671 | -1.7296 | -0.3354 | 0.5833 | -0.3811 |
| phi2 | -0.8808 | -0.8078 | -0.8808 | -0.0883 | 0.1972 | -0.3010 |
| theta1 | 1.6115 | 1.5992 | 1.6115 | 0.2727 | -0.6670 | 0.3090 |
| theta2 | 0.7384 | 0.7352 | 0.7384 | 0.0948 | -0.1490 | 0.2944 |
| Theta1 | -0.0764 | -0.0621 | -0.0764 | -0.0323 | -0.0089 | -0.0111 |
| delta (in-mean) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1825 |
| sigma2 (uncond.) | 1.3531 | 1.7863 | 1.0959 | 0.9310 | 0.9641 | 0.8395 |
| omega | 1.3531 | 1.7863 | 0.0435 | 0.0440 | 0.0400 | 0.0502 |
| alpha | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gamma | 0.0000 | 0.0000 | 0.2306 | 0.2447 | 0.2122 | 0.2103 |
| b | 0.0000 | 0.0000 | 0.8450 | 0.8304 | 0.8524 | 0.8351 |
| persistence | 0.0000 | 0.0000 | 0.9603 | 0.9527 | 0.9585 | 0.9402 |
| nu | inf | 2.7189 | 5.7790 | 5.2028 | 6.0499 | 6.0066 |
| lambda (skew) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.2606 | -0.2494 |

## Validation, 1 day ahead

| validation, 1 day ahead (fixed origin) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| NLL / day | 2.083 | 2.08 | 1.856 | 1.846 | 1.835 | 1.841 |
| Ljung-Box z (10) p | 0.168 | 0.6546 | 0.1475 | 0.8276 | 0.9162 | 0.8289 |
| Ljung-Box z^2 (10) p | 1.708e-10 | 5.347e-12 | 0.3424 | 0.3666 | 0.443 | 0.3819 |
| ARCH-LM z (5) p | 0.0003925 | 8.209e-05 | 0.1863 | 0.185 | 0.2428 | 0.2351 |

## Test, 1 day ahead (exact density)

| test, 1 day ahead (exact density) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| NLL / day | 1.668 | 1.587 | 1.564 | 1.543 | 1.539 | 1.539 |
| Ljung-Box z (10) p | 0.0273 | 0.211 | 0.1045 | 0.9111 | 0.9314 | 0.9169 |
| Ljung-Box z^2 (10) p | 1.203e-27 | 7.141e-27 | 0.8633 | 0.8425 | 0.8671 | 0.8426 |
| ARCH-LM z (5) p | 1.107e-18 | 1.823e-19 | 0.5255 | 0.4636 | 0.5044 | 0.4561 |

## Test, 1 day ahead (cumulative log return, %)

| test, 1 day ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 0.6839 | 0.671 | 0.6725 | 0.6586 | 0.6575 | 0.6582 |
| MSE | 1.667 | 1.637 | 1.664 | 1.623 | 1.622 | 1.623 |
| MSE, zero forecast | 1.637 | 1.637 | 1.637 | 1.637 | 1.637 | 1.637 |
| 90% interval width | 4.164 | 3.8 | 3.73 | 3.669 | 3.631 | 3.615 |
| cover 90% | 0.9208 | 0.8852 | 0.8661 | 0.8716 | 0.8593 | 0.8689 |
| cover 95% | 0.9577 | 0.959 | 0.9413 | 0.9372 | 0.929 | 0.9303 |
| VaR 5% hit rate | 0.04372 | 0.06831 | 0.06967 | 0.0765 | 0.06011 | 0.05874 |
| VaR 1% hit rate | 0.01913 | 0.009563 | 0.01776 | 0.02186 | 0.01639 | 0.01639 |
| VaR 5% Kupiec p | 0.4258 | 0.03092 | 0.02082 | 0.002193 | 0.2232 | 0.2903 |
| VaR 5% indep. p | 0.05573 | 0.3864 | 0.09085 | 0.1877 | 0.6592 | 0.2505 |
| VaR 1% Kupiec p | 0.02751 | 0.9047 | 0.05718 | 0.005319 | 0.1115 | 0.1115 |

## Test, 3 days ahead (cumulative log return, %)

| test, 3 days ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 1.169 | 1.151 | 1.136 | 1.126 | 1.128 | 1.126 |
| MSE | 4.687 | 4.614 | 4.66 | 4.602 | 4.618 | 4.583 |
| MSE, zero forecast | 4.726 | 4.726 | 4.726 | 4.726 | 4.726 | 4.726 |
| 90% interval width | 6.921 | 6.933 | 6.241 | 6.235 | 6.071 | 6.011 |
| cover 90% | 0.9098 | 0.9071 | 0.873 | 0.8634 | 0.8525 | 0.8538 |
| cover 95% | 0.9508 | 0.9604 | 0.9522 | 0.9467 | 0.9372 | 0.9303 |
| VaR 5% hit rate | 0.04645 | 0.05328 | 0.06421 | 0.08607 | 0.07514 | 0.07787 |
| VaR 1% hit rate | 0.01366 | 0.008197 | 0.0123 | 0.01366 | 0.009563 | 0.008197 |

## Test, 1 week ahead (cumulative log return, %)

| test, 1 week ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 1.478 | 1.454 | 1.43 | 1.418 | 1.421 | 1.419 |
| MSE | 7.174 | 7.037 | 7.149 | 7.034 | 7.072 | 6.993 |
| MSE, zero forecast | 7.358 | 7.358 | 7.358 | 7.358 | 7.358 | 7.358 |
| 90% interval width | 8.875 | 9.143 | 7.981 | 7.968 | 7.719 | 7.601 |
| cover 90% | 0.9085 | 0.9098 | 0.8798 | 0.8866 | 0.8661 | 0.8648 |
| cover 95% | 0.9467 | 0.9672 | 0.9495 | 0.9549 | 0.9317 | 0.9331 |
| VaR 5% hit rate | 0.04645 | 0.05464 | 0.05191 | 0.06284 | 0.04918 | 0.05328 |
| VaR 1% hit rate | 0.009563 | 0.009563 | 0.008197 | 0.005464 | 0.002732 | 0.004098 |

## Test, 2 weeks ahead (cumulative log return, %)

| test, 2 weeks ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 2.027 | 1.975 | 1.96 | 1.922 | 1.937 | 1.931 |
| MSE | 13.39 | 12.99 | 13.37 | 13.01 | 13.17 | 12.88 |
| MSE, zero forecast | 14.13 | 14.13 | 14.13 | 14.13 | 14.13 | 14.13 |
| 90% interval width | 12.27 | 12.86 | 10.95 | 10.94 | 10.64 | 10.33 |
| cover 90% | 0.9153 | 0.9262 | 0.8743 | 0.8907 | 0.8675 | 0.8607 |
| cover 95% | 0.9522 | 0.9658 | 0.9454 | 0.9604 | 0.9331 | 0.9344 |
| VaR 5% hit rate | 0.04372 | 0.05055 | 0.05738 | 0.06831 | 0.05738 | 0.06694 |
| VaR 1% hit rate | 0.01503 | 0.0123 | 0.01093 | 0.0123 | 0.006831 | 0.009563 |

## Test, 1 month ahead (cumulative log return, %)

| test, 1 month ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 2.808 | 2.712 | 2.697 | 2.611 | 2.647 | 2.642 |
| MSE | 25.52 | 24.29 | 25.59 | 24.24 | 24.84 | 23.91 |
| MSE, zero forecast | 28.78 | 28.78 | 28.78 | 28.78 | 28.78 | 28.78 |
| 90% interval width | 17.5 | 18.67 | 15.41 | 15.3 | 15.13 | 14.18 |
| cover 90% | 0.9208 | 0.9208 | 0.8934 | 0.9098 | 0.8798 | 0.8702 |
| cover 95% | 0.9454 | 0.959 | 0.9467 | 0.9549 | 0.9399 | 0.9358 |
| VaR 5% hit rate | 0.03962 | 0.05464 | 0.03552 | 0.05191 | 0.03825 | 0.06011 |
| VaR 1% hit rate | 0.02049 | 0.01776 | 0.008197 | 0.0123 | 0.01093 | 0.01366 |

## What each step adds

1-day column: mean log-score gain per day (exact densities). CRPS columns: percentage reduction in CRPS against the previous rung (positive = better). p: Diebold-Mariano with HAC variance (at least h lags for overlapping h-day windows).

| step | comparison | 1d log-score gain / day | p | CRPS gain 1 day (%) | p 1 day | CRPS gain 3 days (%) | p 3 days | CRPS gain 1 week (%) | p 1 week | CRPS gain 2 weeks (%) | p 2 weeks | CRPS gain 1 month (%) | p 1 month |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fat tails | SARIMAX-t vs SARIMAX | 0.081 | 0.14 | 1.88 | 0.00289 | 1.51 | 0.0359 | 1.61 | 0.136 | 2.56 | 0.163 | 3.42 | 0.357 |
| GARCH variance | Two-step vs SARIMAX-t | 0.0229 | 0.169 | -0.213 | 0.8 | 1.25 | 0.243 | 1.64 | 0.145 | 0.778 | 0.612 | 0.571 | 0.853 |
| joint estimation | Joint vs Two-step | 0.0219 | 0.000748 | 2.06 | 0.0021 | 0.936 | 0.0918 | 0.899 | 0.296 | 1.93 | 0.204 | 3.18 | 0.325 |
| skewed errors | Joint skew-t vs Joint | 0.00341 | 0.598 | 0.165 | 0.327 | -0.164 | 0.58 | -0.258 | 0.602 | -0.804 | 0.407 | -1.37 | 0.511 |
| GARCH-in-mean | Full (GARCH-M) vs Joint skew-t | 9.53e-05 | 0.939 | -0.102 | 0.584 | 0.127 | 0.757 | 0.18 | 0.776 | 0.344 | 0.76 | 0.189 | 0.933 |
| total | Full (GARCH-M) vs SARIMAX | 0.129 | 0.0422 | 3.76 | 9.42e-05 | 3.62 | 0.00488 | 4.02 | 0.0109 | 4.75 | 0.0609 | 5.93 | 0.214 |
