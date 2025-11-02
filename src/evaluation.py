import numpy as np
from scipy.stats import norm

def qlike(y, yhat):
    ratio = np.maximum(1e-8, yhat)
    return y/ratio - np.log(y/ratio) - 1

def diebold_mariano(loss_a, loss_b):
    d = np.array(loss_a) - np.array(loss_b)
    dbar = d.mean()
    var = d.var(ddof=1)
    stat = dbar / np.sqrt(var/len(d))
    p = 2*(1 - norm.cdf(abs(stat)))
    return stat, p
