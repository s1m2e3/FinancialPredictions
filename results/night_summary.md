# Night summary

## dd0_top10_alpha_w

| stage | tree | arms | held-out gap (%/yr) | check years: kept tree | vs its start | |
|---|---|---|---|---|---|---|
| 0 | stocks | 0 | +8.070 +- 4.917 | +8.189 | +0.000 +- 0.000 | kept |
| 1 | exposure | 1 | +9.448 +- 4.269 | +9.512 | +1.323 +- 1.150 | kept |
| 2 | stocks | 0 | +7.313 +- 8.029 | +9.512 | +0.000 +- 0.000 | kept |
| 3 | exposure | 1 | +7.313 +- 8.029 | +9.512 | +0.000 +- 0.000 | kept |

**Stock tree**

```
Fallback
\-- buy          # totality guard
```

**Exposure tree**

```
Fallback
|-- Sequence[ spx_dist_200d>0.063 , Prefer(argmax theta z) ]
\-- invest 100%          # totality guard
```

**validation 2020-2021**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                18.608           140584.386         12.613        22.552   1.043         -32.049          0.000    0.000              NaN  1.000       0.000
the 10 largest (by size)                           42.864           203811.505         28.991        27.633   1.572         -25.722          2.581   16.378            0.040  0.992      20.083
stock tree, 100% invested                          42.864           203811.505         28.991        27.633   1.572         -25.722          2.581   16.378            0.040  0.992      20.083
stock tree + exposure tree                         33.387           177717.457         24.990        20.881   1.672         -17.653          6.337   12.377            0.140  0.637      19.909
```

**test 2022-2026**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                 8.498           146506.049          5.444        17.545   0.510         -26.286          0.000    0.000              NaN  1.000       0.000
the 10 largest (by size)                           13.072           177758.889          6.210        26.278   0.571         -38.905          1.327    0.766            0.449  1.374       2.698
stock tree, 100% invested                          13.072           177758.889          6.210        26.278   0.571         -38.905          1.327    0.766            0.449  1.374       2.698
stock tree + exposure tree                          9.879           155448.698          5.045        22.278   0.501         -35.165          4.914   -0.399            0.522  1.051       1.744
```

## dd0.25_w_stocks_core10x50_explore

| stage | tree | arms | held-out gap (%/yr) | check years: kept tree | vs its start | |
|---|---|---|---|---|---|---|
| 0 | stocks | 1 | +4.243 +- 3.803 | +4.171 | +3.276 +- 0.403 | kept |
| 1 | stocks | 1 | +11.700 +- 10.105 | +4.171 | -4.248 +- 4.410 | reverted |

**Stock tree**

```
Fallback
|-- Sequence[ rank_beta_252>0.660 , exit ]
\-- Prefer(argmax theta z)          # totality guard
```

**Exposure tree**

```
Fallback
\-- invest 100%          # totality guard
```

**validation 2020-2021**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                18.608           140584.386         12.613        22.552   1.043         -32.049          0.000    0.000              NaN  1.000       0.000
buy all (by size)                                  29.684           168006.391         20.843        24.264   1.355         -30.111          1.932    8.230            0.106  0.997       9.409
stock tree, 100% invested                          12.843           127275.483          8.363        20.605   0.879         -27.428          4.388   -4.250            0.764  0.856      -2.030
stock tree + exposure tree                         12.843           127275.483          8.363        20.605   0.879         -27.428          4.388   -4.250            0.764  0.856      -2.030
```

**test 2022-2026**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                 8.498           146506.049          5.444        17.545   0.510         -26.286          0.000    0.000              NaN  1.000       0.000
buy all (by size)                                  10.789           161567.463          6.512        20.593   0.565         -30.142          0.772    1.068            0.288  1.152       1.312
stock tree, 100% invested                           5.893           130749.426          3.519        15.826   0.394         -24.045          2.524   -1.925            0.787  0.836      -1.248
stock tree + exposure tree                          5.893           130749.426          3.519        15.826   0.394         -24.045          2.524   -1.925            0.787  0.836      -1.248
```

## dd0.25_top20_w_stocks_core10x50_explore

| stage | tree | arms | held-out gap (%/yr) | check years: kept tree | vs its start | |
|---|---|---|---|---|---|---|
| 0 | stocks | 0 | -1.156 +- 7.034 | +4.318 | -0.561 +- 1.879 | reverted |
| 1 | stocks | 0 | -0.858 +- 4.991 | +4.318 | +0.000 +- 0.000 | kept |

**Stock tree**

```
Fallback
\-- buy          # totality guard
```

**Exposure tree**

```
Fallback
\-- invest 100%          # totality guard
```

**validation 2020-2021**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                18.608           140584.386         12.613        22.552   1.043         -32.049          0.000    0.000              NaN  1.000       0.000
the 20 largest (by size)                           28.642           165321.836         19.774        24.864   1.295         -28.850          3.560    7.161            0.172  0.985       9.035
stock tree, 100% invested                          28.642           165321.836         19.774        24.864   1.295         -28.850          3.560    7.161            0.172  0.985       9.035
stock tree + exposure tree                         28.642           165321.836         19.774        24.864   1.295         -28.850          3.560    7.161            0.172  0.985       9.035
```

**test 2022-2026**

```
                                     real annual return %  real value of $100k  CER real %/yr  annual vol %  Sharpe  max drawdown %  turnover / yr  CER gap  p (CER gap > 0)   beta  alpha %/yr
S&P 500 (SPY, dividends reinvested)                 8.498           146506.049          5.444        17.545   0.510         -26.286          0.000    0.000              NaN  1.000       0.000
the 20 largest (by size)                           11.237           164653.933          7.230        19.709   0.601         -26.911          2.284    1.786            0.266  1.067       2.295
stock tree, 100% invested                          11.237           164653.933          7.230        19.709   0.601         -26.911          2.284    1.786            0.266  1.067       2.295
stock tree + exposure tree                         11.237           164653.933          7.230        19.709   0.601         -26.911          2.284    1.786            0.266  1.067       2.295
```

## placebo: results\placebo\dd0.25_w_stocks_core10x50_explore

# Placebo null distribution (1 replicates, drawdown weight 0.25)

| metric | null mean | null 90th pct | null max | real run | p (null >= real) |
|---|---|---|---|---|---|
| validation gap vs buy all | +0.00 | +0.00 | +0.00 | -12.48 | 1.000 |
| test gap vs buy all | +0.00 | +0.00 | +0.00 | -2.99 | 1.000 |
| check gain (sum over stages) | +0.00 | +0.00 | +0.00 | +3.28 | 0.500 |
| arms (stocks + exposure) | +0.00 | +0.00 | +0.00 | +1.00 | 0.500 |

p is the share of placebo replicates doing at least as well as the real run (with the usual +1): below 0.05 the real run did better than selection luck alone explains.

## stability: results\stability\dd0.25_w_stocks_core10x50_explore

# Stability selection (3 replicates, drawdown weight 0.25)

Share of replicates whose kept tree uses each guard (input and direction) or law input.

## stocks tree: guards

| rule | replicates | share |
|---|---|---|
| (none) | 0 | 0% |

## stocks tree: law inputs

| rule | replicates | share |
|---|---|---|
| (none) | 0 | 0% |

stocks: arms per replicate [0, 0, 0]; empty in 3 of 3

## exposure tree: guards

| rule | replicates | share |
|---|---|---|
| (none) | 0 | 0% |

## exposure tree: law inputs

| rule | replicates | share |
|---|---|---|
| (none) | 0 | 0% |

exposure: arms per replicate [0, 0, 0]; empty in 3 of 3

A rule used in at least half the replicates is worth believing; one used once is noise.
