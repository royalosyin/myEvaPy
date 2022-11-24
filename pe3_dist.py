import numpy as np
from numba import njit
from sam_lm import lmom_ratios
from gam_dist import gam_rl_ppf_pval
from utils import gammaln, quastn, unif_rand, draw_bs_sample

"""
Comments: by Chonghua Yin at Nov/13/2021
    Pearson III distribution is mainly used for 
              hydrological extreme value analysis

    Ported from lmoments3.
    Speed up with numba.njit
    Add bootstrapping for confident intervals
"""


@njit
def pe3_lmom_fit(data):
    """
    Fit a Pearson III distribution function to the given data with lmoments methods.
    :param data: used to calculate the gum distribution parameters, array_like
    :return: Pearson III distribution parameters in a list with the order of loc, scale, skew
    """
    n_min = 5  # the default number of moments
    if len(data) >= n_min:
        moments = lmom_ratios(data, nmom=3)  # pe3 distribution only has 3 parameters

        Small = 1e-6
        # Constants used in Minimax Approx:
        C1 = 0.2906
        C2 = 0.1882
        C3 = 0.0442
        D1 = 0.36067
        D2 = -0.59567
        D3 = 0.25361
        D4 = -2.78861
        D5 = 2.56096
        D6 = -0.77045

        t3 = np.abs(moments[2])
        if moments[1] <= 0 or t3 >= 1:
            raise Exception("L-Moments invalid")

        if t3 <= Small:
            loc = moments[0]
            scale = moments[1] * np.sqrt(np.pi)
            skew = 0
        else:
            if t3 >= (1.0 / 3):
                t = 1 - t3
                alpha = t * (D1 + t * (D2 + t * D3)) / (1 + t * (D4 + t * (D5 + t * D6)))
            else:
                t = 3 * np.pi * t3 * t3
                alpha = (1 + C1 * t) / (t * (1 + t * (C2 + t * C3)))

            rtalph = np.sqrt(alpha)
            beta = np.sqrt(np.pi) * moments[1] * \
                   np.exp(gammaln(alpha) - gammaln(alpha + 0.5))
            loc = moments[0]
            scale = beta * rtalph
            skew = 2 / rtalph
            if moments[2] < 0:
                skew *= -1

        return [loc, scale, skew]

    elif len(data) <= n_min:
        raise Exception("At least 5 data points must be provided.")


@njit
def pe3_ppf_pval(prob, loc, scale, skew):
    """
    Extract an extreme value from a pe3 distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param loc: location parameter of pe3
    :param scale: scale parameter of pe3
    :param skew: location parameter of pe3
    :return: pe3 extreme value with the probability of prob
    """
    SMALL = 1e-6

    if loc <= 0:
        print("Warning! Parameters Invalid loc <= 0.")
        return np.nan

    gamma = skew
    if prob <= 0 or prob >= 1:
        if prob == 0 and gamma > 0:
            quape3 = loc - 2.0 * scale / gamma
            return quape3
        elif prob == 1 and gamma < 0:
            quape3 = loc - 2.0 * scale / gamma
            return quape3
        else:
            print("prob Value Invalid")
            return np.nan

    if np.abs(gamma) < SMALL:
        quape3 = loc + scale * quastn(prob)
        return quape3

    alpha = 4.0 / (gamma * gamma)
    beta = np.abs(0.5 * scale * gamma)
    if gamma > 0:
        quape3 = loc - alpha * beta + gam_rl_ppf_pval(prob, alpha, beta)
    if gamma < 0:
        quape3 = loc + alpha * beta - gam_rl_ppf_pval(1 - prob, alpha, beta)

    return quape3


@njit
def pe3_ppf(probs, loc, scale, skew):
    """
    Extract multiple extreme values defined by probs from a pe3 distribution.
    It is a plural format of pe3_ppf_pval
    :param probs: wanted probabilities related to return years
    :param loc: location parameter of pe3
    :param scale: scale parameter of pe3
    :param skew: location parameter of pe3
    :return: pe3 extreme value with the probability of prob
    """
    nprob = len(probs)
    rvals = np.zeros_like(probs)
    for ip in np.arange(nprob):
        prob = probs[ip]
        rvals[ip] = pe3_ppf_pval(prob, loc, scale, skew)

    return rvals


@njit
def pe3_rvs(loc, scale, skew, size=1000):
    """
    Generate random numbers with a pe3 distribution
    :param loc: loc parameter of pe3
    :param scale: scale parameter of pe3
    :param skew: scale parameter of pe3
    :param size: number of sample
    :return: random values of size from the pe3 parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return pe3_ppf(sam_probs, loc, scale, skew)


@njit
def pe3_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the pe3 distribution
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
            u, d, s = pe3_lmom_fit(sample)
            sam_out[i, :] = pe3_ppf(probs, u, d, s)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.percentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci
