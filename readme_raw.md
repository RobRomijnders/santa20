Playing with [Santa 2020](https://www.kaggle.com/c/santa-2020/overview) challenge on Kaggle.
Together with Vincent Warmerdam on 23 December 2020

Our best submission is in idea02/santa.ipynb and uses decayed Thompson sampling.

# Objective for RL

In the multi armed bandit setting, we aim to get maximal reward:
$q\_\*(a) = E[R_t|A_t=a] $

A natural way to estimate the reward is:

$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i {1}_{A_i=a}  }{\sum_{i=1}^{t-1} {1}_{A_i=a} }$

This estimate gives rise to the update formula for a specific action:
$Q_{n+1} = Q_n + \frac{1}{n} (R_n - Q_n)$

Or when discounting earlier results:
$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i$

## Policies

A greedy policy will take action $A_t = arg \max_a Q_t(a) $

An $\epsilon$ greedy policy will take a random action instead of the greedy action with probability $\epsilon$

## Estimation methods

We can make a running estimate for a stochastic process $x_i$ using:

$\mu_t = \frac{\sum_{i<t} w_i x_i }{\sum_{i<t} w_i}$
$\sigma^2_t = \frac{\sum_{i<t} w_i (x_i - \mu_i)^2 }{\sum_{i<t} w_i}$

Where $w_i$ are arbitrary non-negative weighting coefficients.

# Render latex

Command to render latex in Readme. This uses [readme2tex](https://github.com/leegao/readme2tex)
`python -m readme2tex --branch master --username robromijnders --project santa20 --htmlize --output readme.md readme_raw.md`
