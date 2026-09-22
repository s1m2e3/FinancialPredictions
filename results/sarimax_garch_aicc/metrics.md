# SARIMAX + GARCH, 1 day to 1 month ahead: results

Mean equation from identification.py, lowest AICc: **SARIMA(4,1,4)(2,0,2)[5]**.

| window | dates | days |
|---|---|---|
| train | 2013-01-04 .. 2020-12-31 | 2013 |
| validation | .. 2022-12-31 | 503 |
| test | .. 2025-12-31 | 752 (walk-forward, refit every 63 days, 5000 simulated paths per origin) |

## Parameters (train fit)

| parameter (train fit) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| c | 0.0545 | 0.1390 | 0.0545 | 0.1136 | 0.0694 | -0.0617 |
| phi1 | -0.1359 | 0.0036 | -0.1359 | -0.2390 | -0.1139 | -0.3301 |
| phi2 | 0.8893 | 1.0444 | 0.8893 | -0.4171 | -0.5741 | -0.8218 |
| phi3 | -0.3155 | -0.1627 | -0.3155 | 0.3743 | 0.4136 | 0.1343 |
| phi4 | -0.8877 | -0.7450 | -0.8877 | 0.4757 | 0.2763 | 0.1219 |
| Phi1 | 0.0378 | 0.8331 | 0.0378 | 0.9009 | 1.1362 | 1.1297 |
| Phi2 | -0.2779 | -0.2396 | -0.2779 | -0.2883 | -0.5204 | -0.4842 |
| theta1 | 0.0147 | -0.0757 | 0.0147 | 0.1783 | 0.0361 | 0.2591 |
| theta2 | -0.8440 | -1.0260 | -0.8440 | 0.4186 | 0.5712 | 0.8140 |
| theta3 | 0.4215 | 0.2223 | 0.4215 | -0.4121 | -0.4742 | -0.1918 |
| theta4 | 0.7460 | 0.7114 | 0.7460 | -0.4530 | -0.2538 | -0.1060 |
| Theta1 | -0.0981 | -0.9020 | -0.0981 | -0.9142 | -1.1457 | -1.1384 |
| Theta2 | 0.2833 | 0.2393 | 0.2833 | 0.2683 | 0.5117 | 0.4770 |
| delta (in-mean) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1599 |
| sigma2 (uncond.) | 1.3502 | 1.8440 | 1.1056 | 0.9545 | 0.9892 | 0.8563 |
| omega | 1.3502 | 1.8440 | 0.0454 | 0.0468 | 0.0414 | 0.0506 |
| alpha | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0.0009 | 0.0010 |
| gamma | 0.0000 | 0.0000 | 0.2248 | 0.2508 | 0.2201 | 0.2121 |
| b | 0.0000 | 0.0000 | 0.8465 | 0.8238 | 0.8472 | 0.8338 |
| persistence | 0.0000 | 0.0000 | 0.9589 | 0.9510 | 0.9582 | 0.9409 |
| nu | inf | 2.6663 | 6.3404 | 5.0925 | 5.9823 | 5.9521 |
| lambda (skew) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.2527 | -0.2462 |

## Validation, 1 day ahead

| validation, 1 day ahead (fixed origin) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| NLL / day | 2.09 | 2.075 | 1.864 | 1.843 | 1.832 | 1.837 |
| Ljung-Box z (10) p | 0.2105 | 0.6415 | 0.1656 | 0.9196 | 0.9362 | 0.9044 |
| Ljung-Box z^2 (10) p | 4.167e-11 | 2.492e-11 | 0.4154 | 0.381 | 0.5036 | 0.4659 |
| ARCH-LM z (5) p | 0.0001709 | 5.506e-05 | 0.2879 | 0.154 | 0.2352 | 0.2569 |

## Test, 1 day ahead (exact density)

| test, 1 day ahead (exact density) | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| NLL / day | 1.669 | 1.592 | 1.566 | 1.547 | 1.542 | 1.542 |
| Ljung-Box z (10) p | 0.02714 | 0.1332 | 0.09353 | 0.8766 | 0.9298 | 0.8936 |
| Ljung-Box z^2 (10) p | 5.403e-27 | 1.087e-29 | 0.8806 | 0.8001 | 0.8144 | 0.7936 |
| ARCH-LM z (5) p | 3.646e-18 | 1.053e-20 | 0.573 | 0.4108 | 0.4379 | 0.3995 |

## Test, 1 day ahead (cumulative log return, %)

| test, 1 day ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 0.6849 | 0.6733 | 0.6737 | 0.6606 | 0.6606 | 0.6599 |
| MSE | 1.67 | 1.64 | 1.67 | 1.631 | 1.636 | 1.629 |
| MSE, zero forecast | 1.637 | 1.637 | 1.637 | 1.637 | 1.637 | 1.637 |
| 90% interval width | 4.165 | 3.789 | 3.748 | 3.684 | 3.655 | 3.64 |
| cover 90% | 0.9221 | 0.8893 | 0.873 | 0.873 | 0.8648 | 0.8593 |
| cover 95% | 0.9549 | 0.9563 | 0.9385 | 0.9385 | 0.9317 | 0.9303 |
| VaR 5% hit rate | 0.04235 | 0.06967 | 0.06421 | 0.07377 | 0.05738 | 0.06011 |
| VaR 1% hit rate | 0.02049 | 0.01093 | 0.01639 | 0.02186 | 0.01639 | 0.01776 |
| VaR 5% Kupiec p | 0.3299 | 0.02082 | 0.09049 | 0.005696 | 0.3704 | 0.2232 |
| VaR 5% indep. p | 0.04435 | 0.4347 | 0.1551 | 0.2403 | 0.7719 | 0.6592 |
| VaR 1% Kupiec p | 0.01246 | 0.8035 | 0.1115 | 0.005319 | 0.1115 | 0.05718 |

## Test, 3 days ahead (cumulative log return, %)

| test, 3 days ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 1.175 | 1.145 | 1.145 | 1.127 | 1.13 | 1.126 |
| MSE | 4.723 | 4.538 | 4.72 | 4.597 | 4.619 | 4.574 |
| MSE, zero forecast | 4.726 | 4.726 | 4.726 | 4.726 | 4.726 | 4.726 |
| 90% interval width | 6.912 | 6.835 | 6.247 | 6.262 | 6.114 | 6.055 |
| cover 90% | 0.9071 | 0.9071 | 0.8716 | 0.8648 | 0.8525 | 0.8511 |
| cover 95% | 0.9536 | 0.9658 | 0.9481 | 0.9467 | 0.9372 | 0.9317 |
| VaR 5% hit rate | 0.04918 | 0.05874 | 0.06831 | 0.08333 | 0.07104 | 0.07787 |
| VaR 1% hit rate | 0.01503 | 0.008197 | 0.01366 | 0.0123 | 0.008197 | 0.008197 |

## Test, 1 week ahead (cumulative log return, %)

| test, 1 week ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 1.488 | 1.437 | 1.443 | 1.415 | 1.421 | 1.417 |
| MSE | 7.245 | 6.9 | 7.223 | 7.013 | 7.067 | 6.996 |
| MSE, zero forecast | 7.358 | 7.358 | 7.358 | 7.358 | 7.358 | 7.358 |
| 90% interval width | 8.851 | 8.854 | 7.993 | 7.968 | 7.739 | 7.659 |
| cover 90% | 0.9044 | 0.9085 | 0.8757 | 0.8839 | 0.8702 | 0.8675 |
| cover 95% | 0.9481 | 0.9631 | 0.9481 | 0.9563 | 0.9317 | 0.9344 |
| VaR 5% hit rate | 0.04781 | 0.05601 | 0.04918 | 0.06148 | 0.04645 | 0.05191 |
| VaR 1% hit rate | 0.009563 | 0.009563 | 0.004098 | 0.005464 | 0.002732 | 0.005464 |

## Test, 2 weeks ahead (cumulative log return, %)

| test, 2 weeks ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 2.038 | 1.964 | 1.972 | 1.922 | 1.94 | 1.933 |
| MSE | 13.49 | 12.97 | 13.51 | 12.95 | 13.18 | 12.92 |
| MSE, zero forecast | 14.13 | 14.13 | 14.13 | 14.13 | 14.13 | 14.13 |
| 90% interval width | 12.3 | 12.52 | 11.04 | 10.95 | 10.68 | 10.45 |
| cover 90% | 0.9167 | 0.9208 | 0.8743 | 0.888 | 0.862 | 0.8689 |
| cover 95% | 0.9522 | 0.9658 | 0.9536 | 0.9617 | 0.9385 | 0.9344 |
| VaR 5% hit rate | 0.04372 | 0.06011 | 0.05601 | 0.06557 | 0.05601 | 0.06148 |
| VaR 1% hit rate | 0.01503 | 0.01503 | 0.009563 | 0.01093 | 0.008197 | 0.0123 |

## Test, 1 month ahead (cumulative log return, %)

| test, 1 month ahead | SARIMAX | SARIMAX-t | Two-step | Joint | Joint skew-t | Full (GARCH-M) |
|---|---|---|---|---|---|---|
| CRPS | 2.828 | 2.714 | 2.711 | 2.625 | 2.662 | 2.656 |
| MSE | 25.78 | 24.62 | 25.84 | 24.27 | 24.92 | 24.13 |
| MSE, zero forecast | 28.78 | 28.78 | 28.78 | 28.78 | 28.78 | 28.78 |
| 90% interval width | 17.67 | 17.56 | 15.65 | 14.92 | 15 | 14.25 |
| cover 90% | 0.9249 | 0.9153 | 0.8921 | 0.903 | 0.8784 | 0.8634 |
| cover 95% | 0.9481 | 0.9522 | 0.9467 | 0.9495 | 0.9372 | 0.9358 |
| VaR 5% hit rate | 0.03825 | 0.06557 | 0.03552 | 0.05601 | 0.03825 | 0.06011 |
| VaR 1% hit rate | 0.01503 | 0.02322 | 0.008197 | 0.01503 | 0.01093 | 0.01366 |

## What each step adds

1-day column: mean log-score gain per day (exact densities). CRPS columns: percentage reduction in CRPS against the previous rung (positive = better). p: Diebold-Mariano with HAC variance (at least h lags for overlapping h-day windows).

| step | comparison | 1d log-score gain / day | p | CRPS gain 1 day (%) | p 1 day | CRPS gain 3 days (%) | p 3 days | CRPS gain 1 week (%) | p 1 week | CRPS gain 2 weeks (%) | p 2 weeks | CRPS gain 1 month (%) | p 1 month |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fat tails | SARIMAX-t vs SARIMAX | 0.0764 | 0.148 | 1.69 | 0.0567 | 2.55 | 0.0203 | 3.42 | 0.0264 | 3.61 | 0.152 | 4.03 | 0.426 |
| GARCH variance | Two-step vs SARIMAX-t | 0.026 | 0.159 | -0.0582 | 0.959 | -0.0202 | 0.988 | -0.413 | 0.795 | -0.398 | 0.852 | 0.089 | 0.984 |
| joint estimation | Joint vs Two-step | 0.0192 | 0.0118 | 1.95 | 0.00717 | 1.56 | 0.0405 | 1.94 | 0.0744 | 2.56 | 0.159 | 3.17 | 0.417 |
| skewed errors | Joint skew-t vs Joint | 0.00485 | 0.45 | -0.0011 | 0.996 | -0.245 | 0.407 | -0.429 | 0.365 | -0.957 | 0.309 | -1.38 | 0.523 |
| GARCH-in-mean | Full (GARCH-M) vs Joint skew-t | 0.000204 | 0.86 | 0.113 | 0.495 | 0.309 | 0.365 | 0.271 | 0.625 | 0.383 | 0.701 | 0.224 | 0.91 |
| total | Full (GARCH-M) vs SARIMAX | 0.127 | 0.0433 | 3.66 | 0.000323 | 4.11 | 0.0031 | 4.76 | 0.0052 | 5.17 | 0.0579 | 6.08 | 0.238 |
