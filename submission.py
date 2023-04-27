import numpy as np


def predict_gp(ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    """ Returns the prediction of f*(x) for x = x3, x4, and x5, given data 
    points (x1m y1), (x2, y2), noise level sigma_W, and a squared exponential 
    with length scale parameter ell and initial uncertainty envelope sigma_0
    """

    def k(z1, z2):
        return sigma_0**2*np.exp(-0.5*((z1-z2)**2) / (ell**2))
    
    Kstar = np.array([
        [ k(x3, x1), k(x3, x2) ],
        [ k(x4, x1), k(x4, x2) ],
        [ k(x5, x1), k(x5, x2) ],
    ])

    Kstarstar = np.array([
        [ k(x3, x3), k(x3, x4), k(x3, x5)],
        [ k(x4, x3), k(x4, x4), k(x4, x5)],
        [ k(x5, x3), k(x5, x4), k(x5, x5)],

    ])

    K = np.array([
        [k(x1, x1), k(x1, x2)],
        [k(x2, x1), k(x2, x2)]
    ])
    
    A = K + sigma_W**2*np.eye(2)
    y = np.array([y1, y2])

    mu = Kstar @ (np.linalg.solve(A, y))
    Sigma = Kstarstar - Kstar@(np.linalg.solve(A, Kstar.T))

    return (mu[0], mu[1], mu[2], Sigma[0,0], Sigma[0, 1], Sigma[0,2], 
        Sigma[1,1], Sigma[1,2], Sigma[2,2])


def ucb(c, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    y3, y4, y5, sigma3_2, _, _, sigma4_2, _, sigma5_2 = \
        predict_gp(ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5)
    y = np.array([y3, y4, y5])
    s2 = np.array([sigma3_2, sigma4_2, sigma5_2])
    s = np.sqrt(s2)
    ucb = y + c*s
    i_max = np.argmax(ucb)
    x = [x3, x4, x5]
    return x[i_max], ucb[i_max]


def sample_gp(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    rng = np.random.default_rng(seed=int(seed))

    # Your code below
    mu3, mu4, mu5, s33, s34, s35, s44, s45, s55 = \
        predict_gp(ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5)
    mu = np.array([mu3, mu4, mu5])
    Sigma = np.array([ [s33, s34, s35], [s34, s44, s45], [s35, s45, s55] ])

    sample = rng.multivariate_normal(mu, Sigma)
    return sample.tolist()

def mcei(seed, ell, sigma_0, sigma_W, x1, y1, x2, y2, x3, x4, x5):
    rng = np.random.default_rng(seed=int(seed))
    sample_y_vals = np.zeros((1000, 3))
    for i in range(1000):
        sample_y_vals[i, :] = sample_gp(rng.choice(10000), ell, sigma_0, 
                                        sigma_W, x1, y1, x2, y2, x3, x4, x5)
        
    # Your code below
    y_best = max(y1, y2)
    sample_i_vals = sample_y_vals - y_best
    sample_i_vals[sample_i_vals < 0] = 0
    ei_vals = np.mean(sample_i_vals, axis = 0)
    x = [x3, x4, x5]
    i_max = np.argmax(ei_vals)
    return x[i_max], ei_vals[i_max]


