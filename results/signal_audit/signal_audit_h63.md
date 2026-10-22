# Signal audit

Signal audit: does any input the trees read predict what they are trying to predict?

The tree search judges every idea by one number per portfolio episode, a few hundred
overlapping episodes that hold ~10-20 independent half-years; only a huge effect could pass.
The data hold far more: every month ~116 stocks each have a next-month return. This asks,
input by input, the question the standard way (information coefficients), with the power
those ~19,000 stock-months give:

    STOCK SELECTION   on every decision day (every DECIDE_EVERY trading days) the Spearman
                      rank correlation, across that day's tradable universe, between an input
                      and the stocks' return over the next DECIDE_EVERY days (open to open) --
                      the rank IC; its mean over decision days, and a t-statistic with a
                      Newey-West variance (lag 2: the windows do not overlap, the lag guards
                      against slow regimes)
    MARKET TIMING     for every market input, the correlation over decision days between its
                      value and the S&P 500's (SPY, dividends in) next-window return, with the
                      same t-statistic -- a few hundred observations, so far less power
    MULTIPLE TESTING  some 70 inputs are tested: an input "passes" only if |t| clears the
                      Bonferroni bar (two-sided 5% over all of them), not just 2

Both periods are shown: 2006-2019 (what the trees are trained on) and 2020-2026. A real
signal has the same sign in both. An IC of 0.02-0.05 that holds is what the literature calls
a usable factor; an IC indistinguishable from 0 in 2006-2019 is something no method can learn.

EARNINGS SURPRISES (earnings.py) are audited with the rest when data/earnings.npz exists: they
are candidates, not yet inputs of the trees. They start in 2010-11, so on 2006-2019 their
ICs rest on 2011-2019.

HORIZON. --h 63 asks about the next three months instead of the next window (fundamentals and
momentum act over quarters); decision days stay one window apart, so the three-month returns
overlap and the Newey-West lag grows with the overlap.

Return horizon 63 trading days, decision days every 21; 165 decision days in 2006-2019; 77 inputs tested, Bonferroni bar |t| > 3.41.

## Stock selection: rank IC (input vs next-window return, across the day's universe)

```
                        IC 2006-2019  t 2006-2019  hit 2006-2019  IC 2020-2026  t 2020-2026 passes same_sign
input                                                                                                       
rank_rev_sue                   0.068        2.680          0.636        -0.012       -0.255  |t|>2        NO
rank_rev_sue_10q               0.066        2.593          0.627        -0.008       -0.177  |t|>2        NO
rank_ebtda_roa_3y              0.018        1.824          0.503         0.004        0.177              yes
rank_ebtda_roa                 0.016        1.323          0.509        -0.007       -0.335               NO
rank_sue                       0.023        1.288          0.545         0.010        0.327              yes
COMPOSITE (pre-stated)         0.028        1.271          0.564        -0.000       -0.005               NO
rank_mu_21d                    0.024        1.214          0.533         0.003        0.071              yes
rank_mu_63d                    0.023        1.172          0.533         0.003        0.077              yes
rank_mu_126d                   0.023        1.157          0.533         0.004        0.082              yes
rank_book_to_market           -0.011       -1.105          0.430        -0.012       -0.447              yes
rank_sue_10q                   0.018        0.970          0.564         0.016        0.497              yes
rank_pup_5d                    0.019        0.949          0.582        -0.016       -0.496               NO
log_dollar_vol                -0.012       -0.937          0.442         0.039        1.656               NO
rank_ebtda_margin             -0.010       -0.930          0.412        -0.005       -0.280              yes
rank_ebtda_roa_chg             0.007        0.826          0.527         0.002        0.083              yes
rank_pup_21d                   0.016        0.795          0.570        -0.021       -0.618               NO
rank_pup_63d                   0.016        0.780          0.558        -0.021       -0.642               NO
rank_rev_growth               -0.010       -0.658          0.461         0.003        0.080               NO
insider_buy_90d               -0.006       -0.615          0.470         0.013        0.828               NO
rv_21                         -0.015       -0.591          0.527         0.012        0.298               NO
rank_rv_21                    -0.015       -0.591          0.527         0.012        0.298               NO
insider_buyers_90d            -0.006       -0.566          0.457         0.011        0.666               NO
in_sp500                       0.008        0.529          0.527         0.030        1.134              yes
q05_1d                         0.012        0.414          0.455        -0.017       -0.398               NO
q05_21d                        0.011        0.382          0.448        -0.020       -0.473               NO
q05_10d                        0.011        0.365          0.448        -0.020       -0.471               NO
q05_63d                        0.010        0.359          0.448        -0.021       -0.511               NO
rank_ebtda_yield              -0.004       -0.358          0.503        -0.002       -0.077              yes
q05_126d                       0.010        0.346          0.467        -0.021       -0.508               NO
q05_5d                         0.009        0.322          0.448        -0.020       -0.476               NO
q05_3d                         0.009        0.301          0.448        -0.020       -0.482               NO
young                          0.006        0.272          0.512        -0.007       -0.286               NO
rank_mom_12_1                  0.006        0.249          0.497        -0.011       -0.285               NO
mom_12_1                       0.006        0.249          0.497        -0.011       -0.285               NO
sig_5d                        -0.007       -0.246          0.545         0.020        0.482               NO
sig_10d                       -0.007       -0.241          0.545         0.021        0.482               NO
sig_3d                        -0.007       -0.236          0.545         0.020        0.481               NO
rank_sig_21d                  -0.006       -0.212          0.552         0.021        0.478               NO
sig_21d                       -0.006       -0.212          0.552         0.021        0.478               NO
sig_1d                        -0.006       -0.205          0.545         0.019        0.458               NO
rank_beta_252                 -0.005       -0.165          0.539         0.062        1.175               NO
beta_252                      -0.005       -0.165          0.539         0.062        1.175               NO
sig_63d                       -0.005       -0.154          0.539         0.021        0.474               NO
sig_126d                      -0.003       -0.103          0.539         0.021        0.468               NO
rank_sig_126d                 -0.003       -0.103          0.539         0.021        0.468               NO
dist_52w_high                 -0.002       -0.077          0.448        -0.002       -0.046              yes
rank_dist_52w_high            -0.002       -0.077          0.448        -0.002       -0.046              yes
rev_1m                         0.000        0.023          0.539         0.008        0.343              yes
rank_rev_1m                    0.000        0.023          0.539         0.008        0.343              yes
sue_age                        0.000        0.020          0.545         0.011        0.817              yes
```

## Market timing: correlation (input vs the S&P 500's next-window return)

```
               corr 2006-2019  t 2006-2019  corr 2020-2026  t 2020-2026 passes same_sign
input                                                                                   
spxz_mu_5d              0.135        1.379          -0.134       -0.914               NO
spx_q05_126d           -0.195       -1.269          -0.202       -1.086              yes
spxz_mu_21d             0.142        1.213          -0.073       -0.470               NO
spxz_pup_126d           0.209        1.164           0.008        0.036              yes
spxz_pup_63d            0.167        1.059          -0.016       -0.084               NO
spxz_pup_5d             0.129        1.034          -0.203       -1.618               NO
spx_q05_63d            -0.164       -1.016          -0.315       -1.328              yes
spxz_mu_126d            0.109        0.950           0.110        0.516              yes
spxz_mu_63d             0.101        0.905           0.033        0.165              yes
spxz_pup_21d            0.126        0.854          -0.094       -0.530               NO
spx_sig_126d            0.134        0.851           0.313        1.496              yes
spx_sig_63d             0.080        0.429           0.360        1.694              yes
spx_rv_126              0.076        0.389           0.278        1.319              yes
spx_ret_12m            -0.061       -0.264          -0.172       -1.160              yes
spx_dd_52w             -0.060       -0.247          -0.254       -1.262              yes
spx_sig_3d             -0.055       -0.242           0.372        1.564               NO
spx_q05_1d              0.054        0.239          -0.395       -1.657               NO
spx_sig_1d             -0.051       -0.226           0.380        1.608               NO
spx_sig_5d             -0.048       -0.214           0.375        1.581               NO
spx_q05_3d              0.040        0.179          -0.391       -1.587               NO
spx_sig_10d            -0.034       -0.152           0.380        1.543               NO
spx_q05_21d            -0.028       -0.139          -0.386       -1.541              yes
vix                     0.025        0.121           0.430        1.734              yes
spx_q05_5d              0.026        0.120          -0.377       -1.544               NO
spx_sig_21d            -0.023       -0.106           0.400        1.601               NO
spx_dist_200d          -0.010       -0.042          -0.214       -0.998              yes
spx_q05_10d            -0.008       -0.040          -0.383       -1.494              yes
```

**0 inputs clear the Bonferroni bar on 2006-2019; stock inputs with |t| > 2 there and the same sign on 2020-2026: none.**