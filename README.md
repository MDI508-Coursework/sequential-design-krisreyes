# Sequential Design

All function definitions must be placed in the `submission.py` file provided.
You may use the provided Jupyter notebook to run tests of your implementation,
or invoke `pytest` (either manually or using the testing framework of the online
editor). However, only code in `submission.py` counts for your grade.

## Problem 1: Gaussian Processes

Let $f^\star(x)$ be an unknown response function that we model with a 
Gaussian Process. For this homework, assume the prior mean function of the GP is 
identically 0, and the prior covariance function is given by a 
squared-exponential kernel $k(x, x^\prime)$ with length scale $\ell$ and 
uncertainty $\sigma^2_0$. 

Given $n$ data points consisting of input/output response pairs $(x_i, y_i)$ for $i = 1, ..., n$, assume

$$y_i = f(x_i) + W_i,$$

for iid additive Gaussian noise $W_i \sim N(0, \sigma^2_W)$. Recall that, given
query inputs $x^\star_1, ..., x^\star_m$, according to the GP, the corresponding
$m$ response function values 
$$\mathbf f^\star = (f(x^\star_1), ..., f(x^\star_m)), $$
have a multivariate normal posterior distribution 
$$\mathbf f^\star \sim \mathcal N(\boldsymbol \mu^n, \Sigma^n),$$ 
where

$$\begin{align*}
\boldsymbol \mu^n &= K_\star[K + \sigma^2_WI]^{-1} \mathbf y \\
\Sigma^n &= K_{\star\star} - K_\star[K + \sigma^2_WI]^{-1}K^T_{\star}
\end{align*}
$$

where:
* $K = \left(k(x_i, x_j)\right)$ is an $n \times n$ covariance matrix.
* $K_{\star\star} = \left(k(x^\star_i, x^\star_j)\right)$ is an $m \times m$
    covariance matrix.
* $K_\star = \left(k(x^\star_i, x_j)\right)$ is an $m \times n$ cross-covariance 
    matrix

* $I$ is the $n\times n$ identity matrix

(See [Rasmussen and Williams](https://gaussianprocess.org/gpml/chapters/RW.pdf), 
eqns 2.22 - 2.24 for details.)

### Problem 1a. GP Posteriors
Given:

* hyperparmeter values `ell`, `sigma_0`, and `sigma_W` corresponding to 
    $\ell, \sigma_0, \sigma_W$, 

* data points `x1`, `y1`, `x2`, and `y2` corresponding to $(x_1, y_1), (x_2, y_2)$,
  and

* query points `x3`, `x4`, and `x5` corresponding to 
    $x^\star_1, x^\star_2, x^\star_3$,

define a function

```python
def predict_gp(ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
```
that returns nine numbers (in the following order): the posterior values 
$\mu^2_1, \mu^2_2, \mu^2_3$ and 
$\Sigma^2_{1,1}, \Sigma^2_{1,2}, \Sigma^2_{1,3}, \Sigma^2_{2,2}, \Sigma^2_{2,3}, \Sigma^2_{3,3}$ 
corresponding to the flattened posterior mean $\boldsymbol \mu^2$ and 
upper-triangular entries of the posterior covariance matrix $\Sigma^2$.

### Problem 1b. Upper Confidence Bound Policy

Given parameter $c$ to the UCB policy, with acqusition function:
$$\text{UCB}(x) = \mu^n(x) + c\sqrt{\Sigma^n(x, x)},$$
and variables `ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5` as defined
above, define a function

```python
def ucb(c, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
```

that returns two numbers: the $x$ value among $x_3, x_4, x_5$ that 
the UCB policy would select, along with the corresponding UCB value $UCB(x)$.


## Problem 2: Monte Carlo estimates of look-ahead policies

Recall that look-ahead policies simulate the expected value 
$$\text{Score}(x) = \mathbb E_y[Q(x, y)],$$
of a future score $Q(x, y)$, where the expectation is over the posterior 
distribution of $y$. We can estimate this by **sampling** several different 
values of $y = f(x)$, calculating the corresponding values of $Q(x, y)$,
then averaging these values for a Monte Carlo estimate of the expected value.

### Problem 2a: Sampling from a GP

Given a seed for a Numpy RNG, and other variables as defined above, define 
a function

```python
def sample_gp(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    rng = np.random.default_rng(seed=int(seed))
```
that returns a sample of 
$\mathbf f^\star = (f^\star(x_3), f^\star(x_4), f^\star(x_5))$ from the 
posterior GP. To ensure agreement with the test dataset, you may only call
`rng.multivariate_normal`, and you may only call that function once in your
implementation of `sample_gp`.

### Problem 2b: Monte Carlo Expected Improvement

Recall that the future score for the EI policy is the improvement:

$$Q(x, y) = [y - y_\text{best}]^+.$$

Define a function:

```python
def mcei(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    rng = np.random.default_rng(seed=int(seed))

    # Calculate 1000 samples of f(x3), f(x4), f(x5)
    sample_y_vals = np.zeros((1000, 3))
    for i in range(1000):
        sample_y_vals[i, :] = sample_gp(rng.choice(10000), ell, sigma_0, 
                                        sigma_W, x1, y1, x2, y2, x3, x4, x5)

    ...
```

that takes 1000 samples of the response values 
$\mathbf f^\star = (f^\star(x_3), f^\star(x_4), f^\star(x_5))$,
and estimates the EI value for $x_3, x_4$ and $x_5$ using Monte Carlo averaging.
Your function should then return the $x$ value among $x_3, x_4, x_5$ that the 
EI policy would pick, along with its corresponding MC-estimated EI value.


