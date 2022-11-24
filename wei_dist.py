import numpy as np
from numba import njit
from sam_lm import lmom_ratios
from gev_dist import _pelgev
from utils import unif_rand, draw_bs_sample
"""
Comments: by Chonghua Yin at Nov/14/2021
    All functions are ported from lmoments3.
    Speed up with numba.njit
    Add bootstrapping for confident intervals
    
    Weibull distribution mainly is use for extreme wind.
        It is a special scale of GEV distribution.
        
    Potential issues:
        CI may not work sometimes properly.
        Replace numpy.quantile with nanquatile for a simple solution.
        
"""

@njit
def wei_lmom_fit(data):
    """
    Fit a wei distribution function to the given data with L-moments.
    :param data: used to calculate the wei distribution parameters, array_like
    :return: gev parameters in a list with the order of shape, loc and scale
    """
    n_min = 5  # the default number of moments
    if len(data) >= n_min:
        moments = lmom_ratios(data, nmom=3) # wei with 3 parameters
        if moments[1] <= 0:
            raise Exception("L-Moments invalid")
        else:
            c, loc, scale = _pelgev(moments=[-moments[0], moments[1], -moments[2]])
            delta = 1.0 / c
            beta = scale / c
            return [delta, -loc-beta, beta]

    elif len(data) <= n_min:
        raise Exception("At least 5 data points must be provided.")


@njit
def wei_ppf_pval(prob, c, loc, scale):
    """
    Extract an extreme value from a wei distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param c: shape parameter of wei
    :param loc: location parameter of wei
    :param scale: scale parameter of wei
    :return: wei extreme value with the probability of prob
    """

    if loc <= 0.0 or scale <= 0.0:
        print(f"Warning! Invalid parameters in ppf with loc <= 0.0 or scale <= 0.0")
        return np.nan

    if prob < 0.0 or prob > 1.0 :
        print("prob Value Invalid")
        return np.nan

    x_wei = loc + scale*((-np.log(1.0-prob))**(1.0/c))

    return x_wei


@njit
def wei_ppf(probs, c, loc, scale):
    """
    Extract multiple extreme values defined by probs from a wei distribution.
    It is a plural format of wei_ppf_pval
    :param probs: wanted probabilities related to return years
    :param c:  shape parameter of wei
    :param loc: location parameter of wei
    :param scale: scale parameter of wei
    :return: extreme values with np.array(len(probs)) defined by probs
    """
    nprob = len(probs)
    rvals = np.zeros_like(probs)
    for ip in np.arange(nprob):
        prob = probs[ip]
        rvals[ip] = wei_ppf_pval(prob, c, loc, scale)

    return rvals


@njit
def wei_rvs(c, loc, scale, size=1000):
    """
    Generate random numbers with a wei distribution
    :param c: shape parameter of wei
    :param loc: loc parameter of wei
    :param scale: scale parameter of wei
    :param size: number of sample
    :return: random values of size from the gum parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return wei_ppf(sam_probs, c, loc, scale)


@njit
def wei_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the wei distribution
    :param probs: wanted probabilities related to return years
    :param alpha: significant level, default value of 0.1
    :param size: sampling number to estimate confident intervals
    :return: lower and upper confident intervals
             with np.array in shape(len(probs), 2)

    To do: The functions dose not work sometimes!!!!
    """

    out_ci = np.empty((len(probs), 2))
    sam_out = np.empty((size, len(probs)))
    p_low = 100.0 * alpha / 2
    p_high = 100.0 * (1.0 - alpha / 2)

    for i in range(size):
        try:
            sample = draw_bs_sample(data)
            c, u, d = wei_lmom_fit(sample)
            sam_out[i, :] = wei_ppf(probs, c, u, d)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.nanpercentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci
