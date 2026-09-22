# AICc order vs BIC order

AICc: **SARIMA(4,1,4)(2,0,2)[5]**, BIC: **SARIMA(2,1,2)(0,0,1)[5]**. Same six models, same 732 test origins (2023-2025, walk-forward).

Positive = the BIC order forecasts better. CRPS columns are percentage reductions in CRPS; p-values are Diebold-Mariano with HAC variance (at least h lags).

| model | 1d log-score gain / day | p (log score) | CRPS 1 day (%) | p 1 day | CRPS 3 days (%) | p 3 days | CRPS 1 week (%) | p 1 week | CRPS 2 weeks (%) | p 2 weeks | CRPS 1 month (%) | p 1 month |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SARIMAX | 0.000618 | 0.786 | 0.148 | 0.52 | 0.523 | 0.249 | 0.688 | 0.107 | 0.535 | 0.051 | 0.682 | 0.216 |
| SARIMAX-t | 0.00515 | 0.293 | 0.342 | 0.431 | -0.535 | 0.354 | -1.18 | 0.0962 | -0.555 | 0.541 | 0.0482 | 0.974 |
| Two-step | 0.00199 | 0.532 | 0.188 | 0.498 | 0.746 | 0.147 | 0.891 | 0.0639 | 0.623 | 0.0251 | 0.53 | 0.368 |
| Joint | 0.00466 | 0.0228 | 0.305 | 0.115 | 0.12 | 0.62 | -0.165 | 0.613 | -0.0151 | 0.976 | 0.542 | 0.514 |
| Joint skew-t | 0.00323 | 0.12 | 0.471 | 0.023 | 0.2 | 0.305 | 0.00514 | 0.983 | 0.137 | 0.755 | 0.556 | 0.371 |
| Full (GARCH-M) | 0.00312 | 0.104 | 0.258 | 0.166 | 0.0171 | 0.921 | -0.0864 | 0.711 | 0.0974 | 0.787 | 0.521 | 0.309 |

## Mean forecasts

| MSE of the predictive mean | 1 day | 3 days | 1 week | 2 weeks | 1 month |
|---|---|---|---|---|---|
| SARIMAX (AICc) | 1.670 | 4.723 | 7.245 | 13.491 | 25.782 |
| SARIMAX (BIC) | 1.667 | 4.687 | 7.174 | 13.386 | 25.523 |
| SARIMAX-t (AICc) | 1.640 | 4.538 | 6.900 | 12.967 | 24.615 |
| SARIMAX-t (BIC) | 1.637 | 4.614 | 7.037 | 12.994 | 24.292 |
| Two-step (AICc) | 1.670 | 4.720 | 7.223 | 13.507 | 25.838 |
| Two-step (BIC) | 1.664 | 4.660 | 7.149 | 13.368 | 25.590 |
| Joint (AICc) | 1.631 | 4.597 | 7.013 | 12.946 | 24.267 |
| Joint (BIC) | 1.623 | 4.602 | 7.034 | 13.010 | 24.235 |
| Joint skew-t (AICc) | 1.636 | 4.619 | 7.067 | 13.181 | 24.921 |
| Joint skew-t (BIC) | 1.622 | 4.618 | 7.072 | 13.169 | 24.836 |
| Full (GARCH-M) (AICc) | 1.629 | 4.574 | 6.996 | 12.920 | 24.133 |
| Full (GARCH-M) (BIC) | 1.623 | 4.583 | 6.993 | 12.884 | 23.907 |
| zero forecast | 1.637 | 4.726 | 7.358 | 14.128 | 28.776 |
