import numpy as np
from numba import njit
from sam_lm import lmom_ratios
from utils import unif_rand, draw_bs_sample, gammaln
"""
Comments: by Chonghua Yin at Nov/12/2021
    All functions are ported from lmoments3.
    Speed up with numba.njit
    Add bootstrapping for confident intervals
"""

@njit
def _pelgev(moments):
    # begin to fit gev distribution with moments
    SMALL = 1e-5
    eps = 1e-6
    maxit = 20
    EU = 0.57721566
    DL2 = np.log(2)
    DL3 = np.log(3)
    A0 = 0.28377530
    A1 = -1.21096399
    A2 = -2.50728214
    A3 = -1.13455566
    A4 = -0.07138022
    B1 = 2.06189696
    B2 = 1.31912239
    B3 = 0.25077104
    C1 = 1.59921491
    C2 = -0.48832213
    C3 = 0.01573152
    D1 = -0.64363929
    D2 = 0.08985247

    t3 = moments[2]
    if moments[1] <= 0 or np.abs(t3) >= 1:
        raise Exception("L-Moments Invalid")

    if t3 <= 0:
        g = (A0 + t3 * (A1 + t3 * (A2 + t3 * (A3 + t3 * A4)))) / (1 + t3 * (B1 + t3 * (B2 + t3 * B3)))

        if t3 >= -0.8:
            para3 = g
            gam = np.exp(gammaln(1 + g))
            para2 = moments[1] * g / (gam * (1 - 2 ** -g))
            para1 = moments[0] - para2 * (1 - gam) / g
            para = [para3, para1, para2]
            return para
        elif t3 <= -0.97:
            g = 1 - np.log(1 + t3) / DL2

        t0 = (t3 + 3) * 0.5
        for IT in range(1, maxit):
            x2 = 2 ** -g
            x3 = 3 ** -g
            xx2 = 1 - x2
            xx3 = 1 - x3
            t = xx3 / xx2
            deriv = (xx2 * x3 * DL3 - xx3 * x2 * DL2) / (xx2 ** 2)
            gold = g
            g -= (t - t0) / deriv

            if abs(g - gold) <= eps * g:
                para3 = g
                gam = np.exp(gammaln(1 + g))
                para2 = moments[1] * g / (gam * (1 - 2 ** -g))
                para1 = moments[0] - para2 * (1 - gam) / g
                para = [para3, para1, para2]
                return para
        raise Exception("Iteration has not converged")
    else:
        z = 1 - t3
        g = (-1 + z * (C1 + z * (C2 + z * C3))) / (1 + z * (D1 + z * D2))
        if abs(g) < SMALL:
            para2 = moments[1] / DL2
            para1 = moments[0] - EU * para2
            para = [0, para1, para2]
        else:
            para3 = g
            gam = np.exp(gammaln(1 + g))
            para2 = moments[1] * g / (gam * (1 - 2 ** -g))
            para1 = moments[0] - para2 * (1 - gam) / g
            para = [para3, para1, para2]
        return para


@njit
def gev_lmom_fit(data):
    """
    Fit a gev distribution function to the given data with L-moments.
    :param data: used to calculate the gev distribution parameters, array_like
    :return: gev parameters in a list with the order of shape, loc and scale
    """
    n_min = 5  # the default number of moments
    if len(data) >= n_min:
        moments = lmom_ratios(data, nmom=3) # gev with 3 parameters
        return _pelgev(moments)

    elif len(data) <= n_min:
        raise Exception("At least 5 data points must be provided.")


@njit
def gev_ppf_pval(prob, c, loc=0.0, scale=1.0):
    """
    Extract an extreme value from a gev distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param c: shape parameter of gev
    :param loc: location parameter of gev
    :param scale: scale parameter of gev
    :return: gev extreme value with the probability of prob
    """
    if scale < 0.0:
        return 0.0
    if prob <= 0.0 or prob >= 1.0:
        if prob == 0.0 and c < 0.0:
            return loc + scale / c
        if prob == 1.0 and c > 0.0:
            return loc + scale / c
    y = -np.log(-np.log(prob))
    if c != 0.0:
        y = (1.0 - np.exp(-c * y)) / c
        return loc + scale * y


@njit
def gev_ppf(probs, c, loc=0.0, scale=1.0):
    """
    Extract multiple extreme values defined by probs from a gev distribution.
    It is a plural format of gev_ppf_pval
    :param probs: wanted probabilities related to return years
    :param c: shape parameter of gev
    :param loc: location parameter of gev
    :param scale: scale parameter of gev
    :return: extreme values with np.array(len(probs)) defined by probs
    """
    nprob = len(probs)
    rvals = np.zeros_like(probs)
    for ip in np.arange(nprob):
        prob = probs[ip]
        rvals[ip] = gev_ppf_pval(prob, c, loc, scale)

    return rvals


@njit
def gev_rvs(c, loc=0.0, scale=1.0, size=1000):
    """
    Generate random numbers with a gev distribution
    :param c: shape parameter of gev
    :param loc: location parameter of gev
    :param scale: scale parameter of gev
    :param size: number of sample
    :return: random values of size from the gev parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return gev_ppf(sam_probs, c, loc, scale)


@njit
def gev_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the gev distribution
    :param probs: wanted probabilities related to return years
    :param alpha: significant level, default value of 0.1
    :param size: sampling number to estimate confident intervals
    :return: lower and upper confident intervals
             with np.array in shape(len(probs), 2)
    """

    out_ci = np.empty((len(probs), 2))
    sam_out = np.empty((size, len(probs)))
    p_low = 100.0 * alpha / 2
    p_high = 100.0 * (1.0 - alpha / 2)

    for i in range(size):
        try:
            sample = draw_bs_sample(data)
            c, u, d = gev_lmom_fit(sample)
            sam_out[i, :] = gev_ppf(probs, c, u, d)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.percentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci

