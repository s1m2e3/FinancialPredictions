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

Return horizon 21 trading days, decision days every 21; 167 decision days in 2006-2019; 77 inputs tested, Bonferroni bar |t| > 3.41.

## Stock selection: rank IC (input vs next-window return, across the day's universe)

```
                        IC 2006-2019  t 2006-2019  hit 2006-2019  IC 2020-2026  t 2020-2026 passes same_sign
input                                                                                                       
rank_rev_sue                   0.052        2.911          0.616        -0.000       -0.002  |t|>2        NO
rank_rev_sue_10q               0.049        2.789          0.643         0.001        0.045  |t|>2       yes
COMPOSITE (pre-stated)         0.033        2.203          0.623         0.013        0.540  |t|>2       yes
rank_book_to_market           -0.015       -2.089          0.461        -0.010       -0.643  |t|>2       yes
rank_sue                       0.024        2.009          0.598         0.011        0.777  |t|>2       yes
rank_ebtda_roa_chg             0.014        1.846          0.551         0.002        0.120              yes
rank_mu_21d                    0.023        1.845          0.563         0.008        0.306              yes
rank_mu_63d                    0.023        1.832          0.569         0.008        0.310              yes
rank_mu_126d                   0.023        1.830          0.569         0.008        0.321              yes
rank_sue_10q                   0.022        1.785          0.589         0.018        1.227              yes
rank_pup_63d                   0.018        1.415          0.569        -0.009       -0.471               NO
rank_pup_21d                   0.018        1.410          0.581        -0.008       -0.418               NO
rank_pup_5d                    0.018        1.404          0.575        -0.005       -0.250               NO
rank_ebtda_yield              -0.011       -1.203          0.455        -0.005       -0.244              yes
log_dollar_vol                -0.011       -1.193          0.437         0.024        1.536               NO
rank_ebtda_roa                 0.010        1.191          0.563        -0.008       -0.577               NO
mom_12_1                       0.019        1.151          0.551         0.019        0.804              yes
rank_mom_12_1                  0.019        1.151          0.551         0.019        0.804              yes
sue_age                        0.012        1.130          0.562         0.010        0.718              yes
rank_ebtda_roa_3y              0.007        0.906          0.503         0.004        0.238              yes
rev_1m                        -0.009       -0.669          0.491         0.005        0.205               NO
rank_rev_1m                   -0.009       -0.669          0.491         0.005        0.205               NO
rank_rv_21                    -0.010       -0.551          0.473         0.010        0.356               NO
rv_21                         -0.010       -0.551          0.473         0.010        0.356               NO
q05_126d                       0.010        0.536          0.527        -0.018       -0.593               NO
rank_ebtda_margin             -0.004       -0.503          0.503        -0.003       -0.270              yes
q05_63d                        0.009        0.491          0.533        -0.018       -0.595               NO
q05_21d                        0.008        0.423          0.503        -0.016       -0.531               NO
young                          0.005        0.422          0.500         0.012        0.686              yes
q05_10d                        0.007        0.374          0.497        -0.016       -0.522               NO
in_sp500                       0.004        0.373          0.521         0.016        0.940              yes
q05_1d                         0.006        0.338          0.485        -0.014       -0.473               NO
q05_5d                         0.006        0.327          0.491        -0.016       -0.544               NO
q05_3d                         0.006        0.292          0.485        -0.016       -0.555               NO
rank_rev_growth                0.003        0.242          0.497         0.005        0.198              yes
insider_buy_90d               -0.002       -0.231          0.464         0.007        0.549               NO
sig_3d                        -0.004       -0.206          0.503         0.017        0.553               NO
insider_buyers_90d            -0.002       -0.204          0.464         0.005        0.404               NO
sig_5d                        -0.004       -0.197          0.503         0.016        0.543               NO
sig_10d                       -0.004       -0.184          0.503         0.017        0.551               NO
sig_1d                        -0.003       -0.166          0.509         0.016        0.541               NO
sig_21d                       -0.003       -0.130          0.497         0.017        0.547               NO
rank_sig_21d                  -0.003       -0.130          0.497         0.017        0.547               NO
rank_beta_252                  0.002        0.119          0.509         0.030        0.870              yes
beta_252                       0.002        0.119          0.509         0.030        0.870              yes
sig_63d                       -0.002       -0.089          0.497         0.018        0.584               NO
rank_dist_52w_high             0.001        0.071          0.509         0.022        0.937              yes
dist_52w_high                  0.001        0.071          0.509         0.022        0.937              yes
sig_126d                      -0.001       -0.033          0.497         0.020        0.623               NO
rank_sig_126d                 -0.001       -0.033          0.497         0.020        0.623               NO
```

## Market timing: correlation (input vs the S&P 500's next-window return)

```
               corr 2006-2019  t 2006-2019  corr 2020-2026  t 2020-2026 passes same_sign
input                                                                                   
spxz_mu_63d             0.148        1.809           0.124        1.054              yes
spxz_mu_126d            0.144        1.721           0.139        1.167              yes
spxz_pup_63d            0.171        1.649          -0.005       -0.056               NO
spxz_mu_21d             0.148        1.585           0.165        1.183              yes
spxz_pup_126d           0.186        1.511           0.076        0.792              yes
spxz_mu_5d              0.117        1.479           0.027        0.303              yes
spxz_pup_5d             0.116        1.448          -0.016       -0.203               NO
spxz_pup_21d            0.126        1.309           0.020        0.248              yes
spx_sig_126d            0.101        0.861           0.287        1.884              yes
spx_q05_126d           -0.049       -0.511          -0.133       -1.072              yes
spx_dd_52w             -0.048       -0.298          -0.195       -1.560              yes
spx_q05_63d            -0.031       -0.281          -0.203       -1.309              yes
spx_sig_63d             0.037        0.254           0.219        1.544              yes
spx_dist_200d          -0.037       -0.241          -0.198       -1.511              yes
vix                     0.032        0.222           0.267        1.732              yes
spx_rv_126              0.030        0.210           0.226        1.423              yes
spx_ret_12m            -0.027       -0.178          -0.090       -0.951              yes
spx_q05_3d              0.020        0.130          -0.230       -1.442               NO
spx_q05_1d              0.018        0.119          -0.226       -1.452               NO
spx_sig_1d             -0.010       -0.066           0.216        1.381               NO
spx_sig_10d             0.009        0.061           0.216        1.344              yes
spx_q05_10d            -0.008       -0.058          -0.204       -1.215              yes
spx_sig_3d             -0.007       -0.046           0.214        1.375               NO
spx_sig_21d             0.005        0.033           0.234        1.474              yes
spx_q05_5d              0.004        0.028          -0.228       -1.416               NO
spx_sig_5d             -0.002       -0.011           0.210        1.344               NO
spx_q05_21d            -0.001       -0.004          -0.182       -1.111              yes
```

**0 inputs clear the Bonferroni bar on 2006-2019; stock inputs with |t| > 2 there and the same sign on 2020-2026: ['rank_rev_sue_10q', 'COMPOSITE (pre-stated)', np.str_('rank_book_to_market'), 'rank_sue'].**