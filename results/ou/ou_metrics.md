# OU with jumps vs SARIMAX + GARCH

Compared with the AICc SARIMAX run on the same 732 test origins (2023-2025, walk-forward, quarterly refits).

## OU parameters (train fit)

| parameter (train fit, 2013-2020) | OU + jumps, constant target | OU + jumps, trend target | OU + GARCH + jumps, trend target |
|---|---|---|---|
| phi | 1 | 1 | 1 |
| kappa per day | 1.803e-08 | 7.266e-07 | 2.168e-09 |
| half-life (days) | 3.844e+07 | 9.539e+05 | 3.198e+08 |
| target at start (level) | 5879 | 5872 | 5876 |
| trend (% per year) | 0 | 17.83 | 10.62 |
| diffusion variance | 0.4344 | 0.4323 | 0.3124 |
| GARCH persistence | 0 | 0 | 0.9285 |
| jumps per day | 0.2974 | 0.2916 | 0.1619 |
| mean jump (%) | -0.5648 | -0.4055 | -0.8374 |
| jump s.d. (%) | 1.758 | 1.742 | 0.927 |
| train NLL / day | 1.448 | 1.446 | 1.316 |

## Head to head

Positive = the OU model is better. CRPS columns: percentage reduction in CRPS; p: Diebold-Mariano with HAC variance (at least h lags).

| comparison | 1d log-score gain / day | p | CRPS 1 day (%) | p 1 day | CRPS 3 days (%) | p 3 days | CRPS 1 week (%) | p 1 week | CRPS 2 weeks (%) | p 2 weeks | CRPS 1 month (%) | p 1 month |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OU + jumps, constant target vs SARIMAX | 0.0806 | 0.0932 | 2.59 | 0.000476 | 0.77 | 0.207 | -0.743 | 0.35 | -3.37 | 0.0203 | -7.24 | 0.022 |
| OU + jumps, constant target vs Full (GARCH-M) | -0.0481 | 0.0265 | -1.28 | 0.0774 | -3.18 | 0.0372 | -4.86 | 0.0173 | -8.01 | 0.0326 | -13.6 | 0.0711 |
| OU + jumps, constant target vs GARCH-jump | -0.0563 | 0.00505 | -1.34 | 0.0955 | -2.87 | 0.0132 | -4.3 | 0.000653 | -6.73 | 0.00116 | -11.3 | 0.00447 |
| OU + jumps, trend target vs SARIMAX | 0.0832 | 0.0811 | 2.78 | 0.000368 | 1.54 | 0.00935 | 0.927 | 0.112 | 0.186 | 0.787 | 0.107 | 0.936 |
| OU + jumps, trend target vs Full (GARCH-M) | -0.0455 | 0.0368 | -1.08 | 0.135 | -2.38 | 0.0766 | -3.12 | 0.0419 | -4.29 | 0.096 | -5.78 | 0.245 |
| OU + jumps, trend target vs GARCH-jump | -0.0537 | 0.00809 | -1.14 | 0.179 | -2.07 | 0.0639 | -2.58 | 0.01 | -3.06 | 0.00385 | -3.65 | 0.028 |
| OU + GARCH + jumps, trend target vs SARIMAX | 0.14 | 0.024 | 4.01 | 1.52e-05 | 3.65 | 0.000185 | 3.36 | 0.00024 | 2.9 | 0.00492 | 3.36 | 0.0484 |
| OU + GARCH + jumps, trend target vs Full (GARCH-M) | 0.011 | 0.099 | 0.199 | 0.593 | -0.19 | 0.816 | -0.582 | 0.597 | -1.45 | 0.455 | -2.33 | 0.552 |
| OU + GARCH + jumps, trend target vs GARCH-jump | 0.00272 | 0.529 | 0.146 | 0.708 | 0.11 | 0.86 | -0.0518 | 0.925 | -0.254 | 0.387 | -0.278 | 0.532 |

## Mean forecasts

| MSE of the predictive mean | 1 day | 3 days | 1 week | 2 weeks | 1 month |
|---|---|---|---|---|---|
| SARIMAX | 1.683 | 4.738 | 7.205 | 13.298 | 25.257 |
| Full (GARCH-M) | 1.630 | 4.609 | 7.044 | 12.942 | 24.060 |
| GARCH-jump | 1.627 | 4.665 | 7.178 | 13.387 | 25.748 |
| OU + jumps, constant target | 1.637 | 4.725 | 7.363 | 14.138 | 28.829 |
| OU + jumps, trend target | 1.630 | 4.648 | 7.128 | 13.288 | 25.283 |
| OU + GARCH + jumps, trend target | 1.632 | 4.671 | 7.201 | 13.493 | 26.074 |
| zero forecast | 1.637 | 4.726 | 7.358 | 14.128 | 28.776 |

## OU interval coverage (nominal 90%)

| 90% interval coverage | 1 day | 3 days | 1 week | 2 weeks | 1 month |
|---|---|---|---|---|---|
| OU + jumps, constant target | 0.907 | 0.925 | 0.925 | 0.933 | 0.929 |
| OU + jumps, trend target | 0.910 | 0.918 | 0.921 | 0.932 | 0.933 |
| OU + GARCH + jumps, trend target | 0.865 | 0.874 | 0.884 | 0.893 | 0.900 |
