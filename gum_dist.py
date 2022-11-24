import numpy as np
from numba import njit
from sam_lm import lmom_ratios
from utils import unif_rand, draw_bs_sample
"""
Comments: by Chonghua Yin at Nov/12/2021
    Gumbel distribution is mainly used for 
              wind speed extreme value analysis
              
    Ported from lmoments3.
    Speed up with numba.njit
    Add bootstrapping for confident intervals
"""


@njit
def gum_lmom_fit(data):
    """
    Fit a gumbel distribution function to the given data with lmoments methods.
    :param data: used to calculate the gum distribution parameters, array_like
    :return: gumbel distribution parameters in a list with the order of loc and scale
    """
    n_min = 5  # the default number of moments
    if len(data) >= n_min:
        moments = lmom_ratios(data, nmom=2)  # gumbel distribution only has two parameters

        eu = 0.577215664901532861
        if moments[1] <= 0:
            raise Exception("L-Moments invalid")
        else:
            para2 = moments[1] / np.log(2)
            para1 = moments[0] - eu * para2
            para = [para1, para2]
            return para

    elif len(data) <= n_min:
        raise Exception("At least 5 data points must be provided.")


@njit
def gum_ppf_pval(prob, loc=0.0, scale=1.0):
    """
    Extract an extreme value from a gumbel distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param loc: location parameter of gumbel
    :param scale: scale parameter of gumbel
    :return: gumbel extreme value with the probability of prob
    """
    if scale <= 0:
        print("Warning! Parameters Invalid scale <= 0")
        return np.nan
    if prob <= 0 or prob >= 1:
        print("prob Value Invalid")
        return np.nan
    pval = loc - scale * np.log(-np.log(prob))
    return pval


@njit
def gum_ppf(probs, loc=0.0, scale=1.0):
    """
    Extract multiple extreme values defined by probs from a gum distribution.
    It is a plural format of gum_ppf_pval
    :param probs: wanted probabilities related to return years
    :param loc: location parameter of gum
    :param scale: scale parameter of gum
    :return: extreme values with np.array(len(probs)) defined by probs
    """
    nprob = len(probs)
    rvals = np.zeros_like(probs)
    for ip in np.arange(nprob):
        prob = probs[ip]
        rvals[ip] = gum_ppf_pval(prob, loc, scale)

    return rvals


@njit
def gum_rvs(loc=0.0, scale=1.0, size=1000):
    """
    Generate random numbers with a gum distribution
    :param loc: loc parameter of gum
    :param scale: scale parameter of gum
    :param size: number of sample
    :return: random values of size from the gum parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return gum_ppf(sam_probs, loc, scale)


@njit
def gum_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the gum distribution
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
            u, d = gum_lmom_fit(sample)
            sam_out[i, :] = gum_ppf(probs, u, d)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.percentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci

