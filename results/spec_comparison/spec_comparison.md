# AICc order vs BIC order

AICc: **SARIMA(4,1,4)(2,0,2)[5]**, BIC: **SARIMA(2,1,2)(0,0,1)[5]**. Same six models, same 732 test origins (2023-2025, walk-forward).

Positive = the BIC order forecasts better. CRPS columns are percentage reductions in CRPS; p-values are Diebold-Mariano with HAC variance (at least h lags).

| model | 1d log-score gain / day | p (log score) | CRPS 1 day (%) | p 1 day | CRPS 3 days (%) | p 3 days | CRPS 1 week (%) | p 1 week | CRPS 2 weeks (%) | p 2 weeks | CRPS 1 month (%) | p 1 month |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SARIMAX | 0.00487 | 0.0767 | 0.468 | 0.0858 | 0.623 | 0.119 | 0.263 | 0.515 | -0.39 | 0.523 | -0.247 | 0.829 |
| SARIMAX-t | 0.00921 | 0.0294 | 0.738 | 0.0582 | 0.58 | 0.263 | 0.0647 | 0.914 | -0.447 | 0.613 | 0.349 | 0.804 |
| Two-step | 0.00371 | 0.255 | 0.415 | 0.16 | 0.807 | 0.0565 | 0.45 | 0.303 | -0.231 | 0.705 | 0.148 | 0.899 |
| Joint | 0.00852 | 0.0778 | 0.446 | 0.279 | 0.438 | 0.549 | 0.198 | 0.765 | -0.284 | 0.411 | 0.133 | 0.769 |
| Joint skew-t | 0.00568 | 0.228 | 0.584 | 0.123 | 0.507 | 0.43 | 0.253 | 0.682 | -0.185 | 0.542 | 0.0897 | 0.803 |
| Full (GARCH-M) | 0.00542 | 0.211 | 0.442 | 0.187 | 0.507 | 0.385 | 0.368 | 0.496 | 0.0693 | 0.802 | 0.192 | 0.529 |
| GARCH-jump | 0.00414 | 0.378 | 0.353 | 0.324 | 0.35 | 0.581 | 0.0839 | 0.888 | -0.362 | 0.215 | 0.0573 | 0.853 |

## Mean forecasts

| MSE of the predictive mean | 1 day | 3 days | 1 week | 2 weeks | 1 month |
|---|---|---|---|---|---|
| SARIMAX (AICc) | 1.683 | 4.738 | 7.205 | 13.298 | 25.257 |
| SARIMAX (BIC) | 1.667 | 4.687 | 7.174 | 13.386 | 25.523 |
| SARIMAX-t (AICc) | 1.655 | 4.656 | 7.056 | 12.971 | 24.606 |
| SARIMAX-t (BIC) | 1.635 | 4.612 | 7.037 | 12.988 | 24.294 |
| Two-step (AICc) | 1.685 | 4.732 | 7.180 | 13.274 | 25.308 |
| Two-step (BIC) | 1.664 | 4.660 | 7.149 | 13.375 | 25.591 |
| Joint (AICc) | 1.627 | 4.613 | 7.031 | 12.931 | 24.140 |
| Joint (BIC) | 1.624 | 4.609 | 7.042 | 13.016 | 24.224 |
| Joint skew-t (AICc) | 1.631 | 4.656 | 7.136 | 13.223 | 25.065 |
| Joint skew-t (BIC) | 1.623 | 4.622 | 7.086 | 13.224 | 24.950 |
| Full (GARCH-M) (AICc) | 1.630 | 4.609 | 7.044 | 12.942 | 24.060 |
| Full (GARCH-M) (BIC) | 1.620 | 4.582 | 6.994 | 12.865 | 23.884 |
| GARCH-jump (AICc) | 1.627 | 4.665 | 7.178 | 13.387 | 25.748 |
| GARCH-jump (BIC) | 1.622 | 4.625 | 7.125 | 13.406 | 25.654 |
| zero forecast | 1.637 | 4.726 | 7.358 | 14.128 | 28.776 |
