import numpy as np

"""
Throughout this file:

ell, sigma_0: length-scale and uncertainty hyperparameter to a squared 
    exponential kernel function.

sigma_W: noise level (standard deviation of additive gaussian noise)

x1, y1: An input/output response function observation, y1 = f(x1) + W1,
    where W1 is additive gaussian noise, and f is the ground-truth response 
    function.

x2, y2: Another input/output response function observation, y2 = f(x2) + W2,
    where W2 is additive gaussian noise (independent of W1), and f is again the
    ground-truth response function.

x3, x4, x5: query input values.
"""


def predict_gp(ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    """ Returns the posterior mean prediction and covariances for f(x3), f(x4), 
    and f(x5), given observations (x1, y1), (x2, y2) and hyperparameters (ell,
    sigma_0, sigma_W).
    """


def ucb(c, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    """ Given the a UCB parameter C, and GP inputs, determines which of x3, x4
    or x5 should be selected according to the UCB policy and its associated UCB
    value. """

    # Your code below

def sample_gp(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    """ Given an seed to a Numpy RNG, and GP inputs, provide posterior samples
    of the response function values f(x3, f(x4), f(x5)"""

    # Instantiate the RNG as below
    rng = np.random.default_rng(seed=int(seed))

    # Your code below -- you may use rng.multivariate_normal only ONCE


def mcei(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    """ Given a seed to a Numpy RNG and GP inputs, determine which of x3, x4, 
    x5 would be selected according to the EI policy by calculating the EI
    values using monte carlo averaging. Return the selected x value, along with
    its associated EI value"""

    rng = np.random.default_rng(seed=int(seed))

    # Calculate 1000 samples of f(x3), f(x4), f(x5)
    sample_y_vals = np.zeros((1000, 3))
    for i in range(1000):
        sample_y_vals[i, :] = sample_gp(rng.choice(10000), ell, sigma_0, 
                                        sigma_W, x1, y1, x2, y2, x3, x4, x5)
        
    # Your code below













    



