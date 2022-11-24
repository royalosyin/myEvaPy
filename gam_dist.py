import numpy as np
from deprecated import deprecated
from numba import njit, guvectorize
from rvlib import Gamma

from sam_lm import lmom_ratios
from utils import unif_rand, draw_bs_sample, gammain, \
    gammaln, quastn, digamma, trigamma

"""
Comments: by Chonghua Yin at Nov/12/2021
    1. The gamma parameter estimation, 
        small samples use lmoments of gam_lmom_fit
        large samples use fit_gamma_mle
    2. rvs and ppf, and ci 
        using rvlib version, defined by gam_rl_****
        lmoments version could be used, similar results as rl version, 
                 defined by gam_**** 
"""

@njit
def gam_lmom_fit(data):
    """
    Fit a gamma distribution function to the given data with lmoments methods.
    :param data: used to calculate the gam distribution parameters, array_like
    :return: gamma distribution parameters in a list with the order of loc and scale
    """

    n_min = 5  # the default number of moments
    if len(data) >= n_min:
        moments = lmom_ratios(data, nmom=2)  # gam distribution only has two parameters

        a1 = -0.3080
        a2 = -0.05812
        a3 = 0.01765
        b1 = 0.7213
        b2 = -0.5947
        b3 = -2.1817
        b4 = 1.2113

        if moments[0] <= moments[1] or moments[1] <= 0:
            raise Exception("L-Moments invalid")
        else:
            cv = moments[1] / moments[0]
            if cv >= 0.5:
                t = 1 - cv
                alpha = t * (b1 + t * b2) / (1 + t * (b3 + t * b4))
            else:
                t = np.pi * cv ** 2
                alpha = (1 + a1 * t) / (t * (1 + t * (a2 + t * a3)))

            para = [alpha, moments[0] / alpha]
            return para

    elif len(data) <= n_min:
        raise Exception("At least 5 data points must be provided.")


@njit
def gam_rl_ppf_pval(prob, alpha, beta):
    """
    Extract an extreme value from a gamma distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param alpha: alpha parameter of gamma
    :param beta: beta parameter of gamma
    :return: gamma extreme value with the probability of prob
    """
    return Gamma(alpha, beta).quantile(prob)


@njit
def gam_rl_ppf(probs, alpha, beta):
    """
    Extract multiple extreme values defined by probs from a gum distribution.
    It is a plural format of gum_ppf_pval
    :param probs: wanted probabilities related to return years
    :param loc: location parameter of gum
    :param scale: scale parameter of gum
    :return: extreme values with np.array(len(probs)) defined by probs
    """
    return Gamma(alpha, beta).quantile(probs)


@njit
def gam_rl_rvs(alpha, beta, size=1000):
    """
    Generate random numbers with a gamma distribution
    :param alpha: loc parameter of gam
    :param beta: scale parameter of gam
    :param size: number of sample
    :return: random values of size from the gam parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return gam_rl_ppf(sam_probs, alpha, beta)


@njit
def gam_rl_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the gam distribution
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
            u, d = gam_lmom_fit(sample)
            sam_out[i, :] = gam_rl_ppf(probs, u, d)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.percentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci


@deprecated(reason="we can use rvlib.Gamma!")
@njit
def gam_ppf_pval(prob, alpha, beta):
    """
    Extract an extreme value from a gamma distribution with a probability defined by prob
    :param prob: cdf probability (0.0 - 1.0)
    :param alpha: alpha parameter of gamma
    :param beta: beta parameter of gamma
    :return: gamma extreme value with the probability of prob
    """
    eps = 1e-10
    maxit = 10000

    if alpha <= 0 or beta <= 0:
        print("Warning! Parameters Invalid alpha <= 0 or beta <= 0")
        return np.nan

    if prob <= 0 or prob >= 1:
        print("prob Value Invalid")
        return np.nan

    am1 = alpha - 1.0
    if am1 != 0:
        dlogg = gammaln(alpha)
        if am1 <= 0:
            root = np.exp((np.log(alpha * prob) + dlogg) / alpha)
        else:
            root = alpha * (1. - 1. / (9. * alpha) + quastn(prob) / np.sqrt(9. * alpha)) ** 3.0

        if root <= 0.01 * alpha:
            root = np.exp((np.log(alpha * prob) + dlogg) / alpha)

        for it in np.arange(maxit):
            # func = gammainc_l(root, alpha, True) - prob
            func = gammain(root, alpha) - prob
            rinc = func * np.exp(dlogg + root - am1 * np.log(root))
            root = root - rinc
            if np.abs(func) <= eps:
                quagam = root * beta
                return quagam
    else:
        quagam = -np.log(1.0 - prob) * beta
        return quagam

    print("Result failed to converge")
    return np.nan


@deprecated(reason="we can use rvlib.Gamma!")
@njit
def gam_ppf(probs, loc, scale):
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
        rvals[ip] = gam_ppf_pval(prob, loc, scale)

    return rvals


@deprecated(reason="we can use rvlib.Gamma!")
@njit
def gam_rvs(alpha, beta, size=1000):
    """
    Generate random numbers with a gamma distribution
    :param alpha: loc parameter of gam
    :param beta: scale parameter of gam
    :param size: number of sample
    :return: random values of size from the gam parameters
    """
    sam_probs = unif_rand(a=0., b=1.0, size=size)
    return gam_ppf(sam_probs, alpha, beta)


@deprecated(reason="we can use rvlib.Gamma!")
@njit
def gam_bs_reps_ci(data, probs, alpha=0.1, size=1000):
    """
    Draw boostrap replicates of the confident intervals (alpha=0.1) from 1D data set.
    :param data: raw data used to fit the gam distribution
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
            u, d = gam_lmom_fit(sample)
            sam_out[i, :] = gam_ppf(probs, u, d)
        except:
            pass

    for iprob in range(len(probs)):
        out_ci[iprob, :] = np.percentile(sam_out[:, iprob], [p_low, p_high])

    return out_ci


@njit
def update_gamma_mle(logmx, mlogx, a):
    """
    Update gamma parameter of alpha for each iteration
    :param logmx: input to calculate alpha
    :param mlogx: input to calculate alpha
    :param a: old alpha from last iteration
    :return: new alpha value updated from this iteration
    """
    ia = 1 / a
    z = ia + (mlogx - logmx + np.log(a) - digamma(a)) / (np.power(a, 2) * (ia - trigamma(a)))
    return 1 / z


@njit
def fit_gamma_mle(x:np.array, alpha0=np.nan, maxiter=1000, tol=1e-16):
    """
    The *Gamma distribution with shape parameter `α` and scale `θ`
         has probability density function.
    See more https://en.wikipedia.org/wiki/Gamma_distribution

    Note:
        The accuracy of parameters estimated from MLE depends on the number of samples.
        The more, the accurate!
    :param x: inputs, numpy.array (x>0)
    :param alpha0: initial value for alpha. Default value is np.nan
    :param maxiter: the number of maximum iterations
    :param tol: tolerance between new alpha and old alpha for iteration precision
    :return: shape parameter of alpha and scale parameter of theta
    """
    # check input of x > 0.0
    if np.any(x <= 0):
        return np.nan, np.nan

    # start to work
    sx = np.sum(x)
    slogx = np.sum(np.log(x))
    nrec = len(x)

    mx = sx/nrec
    logmx = np.log(mx)
    mlogx = slogx/nrec

    # iterate to calculate alpha and scale of theta
    if np.isnan(alpha0):
        a = (logmx - mlogx)/2.0
    else:
        a = alpha0

    converged = False
    t = 0.0
    a_old = 0.0
    while (not converged) and (t<maxiter):
        t += 1
        a_old = a
        a = update_gamma_mle(logmx, mlogx, a)
        converged = np.abs(a - a_old) <= tol

    return a, mx/a


@guvectorize(
    ["float64[:], float64[:], float64[:]"],
    "(n), (k) -> (k)",
    nopython=True,target="cpu"
)
def gfunc_fit_gamma(data, dummy_array, out):
    try:
        new_data = data[data>0]
        if len(new_data)<10:
            out[0] = np.nan
            out[1] = np.nan
        else:
            alpha, theda = fit_gamma_mle(new_data)
            out[0] = alpha
            out[1] = theda
    except:
        out[0] = np.nan
        out[1] = np.nan
