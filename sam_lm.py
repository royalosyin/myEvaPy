import numpy as np
from numba import njit
from utils import comb_exact


@njit
def lmom_ratios(x, nmom=5):
    """
    Estimate `nmom` number of L-moments from a sample `data`.
    :param data: Sequence of (sample) data
    :type data: list or array-like sequence
    :param nmom: number of L-moments to estimate
    :type nmom: int
    :return: L-moment ratios like this: l1, l2, t3, t4, t5, .. . As in: items 3 and higher are L-moment ratios.
    :rtype: list
    """
    if nmom > 5:
        raise Exception("Number of lmoments should be less than 5.")

    try:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        x.sort()
    except Exception:
        raise Exception("Input data to estimate L-moments must be numeric.")

    if nmom <= 0 or nmom > 5:
        raise Exception("Invalid number of sample L-moments")

    if n < nmom:
        raise Exception("Insufficient length of data for specified nmoments")

    # First L-moment
    l1 = np.sum(x) / comb_exact(n, 1)
    if nmom == 1:
        return np.array([l1])

    # Second L-moment
    comb1 = np.arange(n)
    coefl2 = 0.5 / comb_exact(n, 2)

    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans = sum_xtrans + (comb1[i] - comb1[n - i - 1]) * x[i]

    l2 = coefl2 * sum_xtrans
    if nmom == 2:
        return np.array([l1, l2])

    # Third L-moment
    comb3 = []
    for i in np.arange(n):
        comb3.append(comb_exact(i, 2))
    coefl3 = 1.0 / 3.0 / comb_exact(n, 3)

    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans += (comb3[i] - 2 * comb1[i] * comb1[n - i - 1] + comb3[n - i - 1]) * x[i]

    l3 = coefl3 * sum_xtrans / l2
    if nmom == 3:
        return np.array([l1, l2, l3])

    # Fourth L-moment
    comb5 = []
    for i in np.arange(n):
        comb5.append(comb_exact(i, 3))
    coefl4 = 0.25 / comb_exact(n, 4)

    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans += (comb5[i] - 3 * comb3[i] * comb1[n - i - 1] + 3 * comb1[i] * comb3[n - i - 1] - comb5[
            n - i - 1]) * x[i]

    l4 = coefl4 * sum_xtrans / l2
    if nmom == 4:
        return np.array([l1, l2, l3, l4])


    # Fifth L-moment
    comb7 = []
    for i in np.arange(n):
        comb7.append(comb_exact(i, 4) )
    coefl5 = 0.2 / comb_exact(n, 5)

    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans += (comb7[i] - 4 * comb5[i] * comb1[n - i - 1] + 6 * comb3[i] * comb3[n - i - 1] -
              4 * comb1[i] * comb5[n - i - 1] + comb7[n - i - 1]) * x[i]


    l5 = coefl5 * sum_xtrans / l2
    return np.array([l1, l2, l3, l4, l5])


@njit
def lmom_ratios_depricated(x, nmom=5):
    """
    Estimate `nmom` number of L-moments from a sample `data`.
    :param x: Sequence of (sample) data
    :type x: list or array-like sequence
    :param nmom: number of L-moments to estimate
    :type nmom: int
    :return: L-moment ratios like this: l1, l2, t3, t4, t5, .. . As in: items 3 and higher are L-moment ratios.
    :rtype: np.array(list)
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        x.sort()
    except Exception:
        raise Exception("Input data to estimate L-moments must be numeric.")

    if nmom <= 0 or nmom > 5:
        raise Exception("Invalid number of sample L-moments")

    if n < nmom:
        raise Exception("Insufficient length of data for specified nmoments")

    # First L-moment
    l1 = np.sum(x) / comb_exact(n, 1)

    if nmom == 1:
        return np.array([l1])

    # Second L-moment
    comb1 = np.arange(n)
    coefl2 = 0.5 / comb_exact(n, 2)
    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans = sum_xtrans + (comb1[i] - comb1[n - i - 1]) * x[i]
    l2 = coefl2 * sum_xtrans

    if nmom == 2:
        return np.array([l1, l2])

    # Third L-moment
    comb3 = np.zeros(n)
    for i in np.arange(n):
        comb3[i] = comb_exact(i, 2)
    coefl3 = 1.0 / 3.0 / comb_exact(n, 3)
    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans = sum_xtrans + (comb3[i] - 2 * comb1[i] * comb1[n - i - 1] + comb3[n - i - 1]) * x[i]
    l3 = coefl3 * sum_xtrans / l2

    if nmom == 3:
        return np.array([l1, l2, l3])

    # Fourth L-moment
    comb5 = np.zeros(n)
    for i in np.arange(n):
        comb5[i] = comb_exact(i, 3)
    coefl4 = 0.25 / comb_exact(n, 4)
    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans = sum_xtrans + (comb5[i] - 3 * comb3[i] * comb1[n - i - 1] +
                                   3 * comb1[i] * comb3[n - i - 1] - comb5[n - i - 1]) * x[i]

    l4 = coefl4 * sum_xtrans / l2

    if nmom == 4:
        return np.array([l1, l2, l3, l4])

    # Fifth L-moment
    comb7 = np.zeros(n)
    for i in np.arange(n):
        comb7[i] = comb_exact(i, 4)
    coefl5 = 0.2 / comb_exact(n, 5)
    sum_xtrans = 0.0
    for i in np.arange(n):
        sum_xtrans = sum_xtrans + (comb7[i] - 4 * comb5[i] * comb1[n - i - 1] +
                                   6 * comb3[i] * comb3[n - i - 1] -
                                   4 * comb1[i] * comb5[n - i - 1] +
                                   comb7[n - i - 1]) * x[i]

    l5 = coefl5 * sum_xtrans / l2

    return np.array([l1, l2, l3, l4, l5])

