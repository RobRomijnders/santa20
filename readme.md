Playing with [Santa 2020](https://www.kaggle.com/c/santa-2020/overview) challenge on Kaggle.

# Objective for RL

In the multi armed bandit setting, we aim to get maximal reward:
<img alt="$q\_\*(a) = E[R_t|A_t=a] $" src="svgs/cf48b34126f164a867332c61136fabfa.svg" align="middle" width="151.011465pt" height="24.6576pt"/>

A natural way to estimate the reward is:

<img alt="$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i {1}_{A_i=a}  }{\sum_{i=1}^{t-1} {1}_{A_i=a} }$" src="svgs/88ce1d040798eda7e8d47c04e1b88438.svg" align="middle" width="151.1796pt" height="40.41708pt"/>

This estimate gives rise to the update formula for a specific action:
<img alt="$Q_{n+1} = Q_n + \frac{1}{n} (R_n - Q_n)$" src="svgs/c6284d6b48408d2f8f56b18cdbc004b3.svg" align="middle" width="190.859955pt" height="27.77577pt"/>

Or when discounting earlier results:
<img alt="$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i$" src="svgs/08444ec4f85339a8b2f88cf8ea577c32.svg" align="middle" width="307.048005pt" height="27.159pt"/>

## Policies

A greedy policy will take action <img alt="$A_t = arg \max_a Q_t(a) $" src="svgs/561f46fdf9ac493b8aa7f5e3076249e8.svg" align="middle" width="149.309655pt" height="24.6576pt"/>

An <img alt="$\epsilon$" src="svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg" align="middle" width="6.6724515pt" height="14.15535pt"/> greedy policy will take a random action instead of the greedy action with probability <img alt="$\epsilon$" src="svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg" align="middle" width="6.6724515pt" height="14.15535pt"/>

## Estimation methods

We can make a running estimate for a stochastic process <img alt="$x_i$" src="svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg" align="middle" width="14.045955pt" height="14.15535pt"/> using:

<img alt="$\mu_t = \frac{\sum_{i&lt;t} w_i x_i }{\sum_{i&lt;t} w_i}$" src="svgs/77a0144ac85bf3048fc605510f7fe3b1.svg" align="middle" width="101.76408pt" height="36.50856pt"/>
<img alt="$\sigma^2_t = \frac{\sum_{i&lt;t} w_i (x_i - \mu_i)^2 }{\sum_{i&lt;t} w_i}$" src="svgs/8690bcffe5346f8bb0d5e46f79ef390d.svg" align="middle" width="143.592735pt" height="39.76302pt"/>

Where <img alt="$w_i$" src="svgs/c2a29561d89e139b3c7bffe51570c3ce.svg" align="middle" width="16.41948pt" height="14.15535pt"/> are arbitrary non-negative weighting coefficients.

# Render latex

Command to render latex in Readme. This uses [readme2tex](https://github.com/leegao/readme2tex)
`python -m readme2tex --branch master --username robromijnders --project santa20 --htmlize --output readme.md readme_raw.md`
