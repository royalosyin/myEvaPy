import numpy as np
from numba import njit
from rvlib import Uniform
from deprecated import deprecated

EPSILON = 1e-37

"""
Comments: by Chonghua Yin at Nov/12/2021
    This section provides some special functions for other modules,
    mainly for *gamma* functions and *random sampling*.
    
    Speed up with numba.njit.
"""


@njit
def gamma(x):
    """
    This routine calculates the GAMMA function for a real argument X.

    Computation is based on an algorithm outlined in reference 1.
    The program uses rational functions that approximate the GAMMA
    function to at least 20 significant decimal digits.  Coefficients
    for the approximation over the interval (1,2) are unpublished.
    Those for the approximation for X .GE. 12 are from reference 2.
    The accuracy achieved depends on the arithmetic system, the
    compiler, the intrinsic functions, and proper selection of the
    machine-dependent constants.
    :param x: scalor, input
    :return: gamma value
    """

    # initialize constants
    ONE = 1.0
    HALF = 0.5
    TWELVE = 12.0
    TWO = 2.0
    ZERO = 0.0
    SQRTPI = 0.9189385332046727417803297
    PI = 3.1415926535897932384626434

    XBIG, XMININ, EPS = 34.844, 5.88e-39, 1.39e-17
    XINF = 1.70e+38
    P = np.array([-1.71618513886549492533811e+0, 2.47656508055759199108314e+1,
                  -3.79804256470945635097577e+2, 6.29331155312818442661052e+2,
                  8.66966202790413211295064e+2, -3.14512729688483675254357e+4,
                  -3.61444134186911729807069e+4, 6.64561438202405440627855e+4])

    Q = np.array([-3.08402300119738975254353e+1, 3.15350626979604161529144e+2,
                  -1.01515636749021914166146e+3, -3.10777167157231109440444e+3,
                  2.25381184209801510330112e+4, 4.75584627752788110767815e+3,
                  -1.34659959864969306392456e+5, -1.15132259675553483497211e+5])

    C = np.array([-1.910444077728e-03, 8.4171387781295e-04,
                  -5.952379913043012e-04, 7.93650793500350248e-04,
                  -2.777777777777681622553e-03, 8.333333333333333331554247e-02,
                  5.7083835261e-03])

    PARITY = False
    FACT = ONE
    N = 0
    Y = x
    if Y <= ZERO:
        # Argument is negative
        Y = -x
        Y1 = int(Y)
        RES = Y - Y1
        if RES != ZERO:
            if Y1 != int(Y1 * HALF) * TWO:
                PARITY = True
                FACT = -PI / np.sin(PI * RES)
                Y = Y + ONE
            else:
                RES = XINF
                return np.nan

    if Y < EPS:
        if Y >= XMININ:
            RES = ONE / Y
        else:
            return np.nan
    elif Y < TWELVE:
        Y1 = Y
        if Y < ONE:
            Z = Y
            Y = Y + ONE
        else:
            N = int(Y) - 1
            Y = Y - int(N)
            Z = Y - ONE

        XNUM = ZERO
        XDEN = ONE

        for I in np.arange(8):
            XNUM = (XNUM + P[I]) * Z
            XDEN = XDEN * Z + Q[I]

        RES = XNUM / XDEN + ONE
        if Y1 < Y:
            RES = RES / Y1
        elif Y1 > Y:
            for I in np.arange(N):
                RES = RES * Y
                Y = Y + ONE
    else:
        if Y <= XBIG:
            YSQ = Y * Y
            SUM = C[6]
            for I in np.arange(6):
                SUM = SUM / YSQ + C[I]
            SUM = SUM / Y - Y + SQRTPI
            SUM = SUM + (Y - HALF) * np.log(Y)
            RES = np.exp(SUM)
        else:
            return np.nan

    if PARITY:
        RES = -RES

    if FACT != ONE:
        RES = FACT / RES

    return RES


@njit
def gammain(x, p):
    """
    GAMAIN computes the incomplete gamma ratio.
    https://people.math.sc.edu/Burkardt/f77_src/asa032/asa032.f
    """
    MAX_IT = 1000
    acu = 1.0e-08
    oflo = 1.0e+37
    uflo = 1.0e-37
    pn = np.zeros(6)

    if p <= 0.0e+00:
        return 0.0

    if x < 0.0:
        return 0.0

    if x == 0.0e+00:
        return 0.0

    g = gammaln(p)
    if np.isnan(g):
        return 0.0e+00

    arg = p * np.log(x) - x - g

    if arg < np.log(uflo):
        return 0.0e+00

    factor = np.exp(arg)

    #  Calculation by series expansion.
    if x <= 1.0e+00 or x < p:
        gin = 1.0e+00
        term = 1.0e+00
        rn = p

        while term > acu:
            rn = rn + 1.0e+00
            term = term * x / rn
            gin = gin + term

        return gin * factor / p

    # Calculation by continued fraction.
    a = 1.0e+00 - p
    b = a + x + 1.0e+00
    term = 0.0e+00

    pn[0] = 1.0e+00
    pn[1] = x
    pn[2] = x + 1.0e+00
    pn[3] = x * b

    gin = pn[2] / pn[3]

    for it in np.arange(MAX_IT):
        a = a + 1.0e+00
        b = b + 2.0e+00
        term = term + 1.0e+00
        an = a * term
        for i in np.arange(2):
            pn[i + 4] = b * pn[i + 2] - an * pn[i]

        if pn[5] != 0.0e+00:
            rn = pn[4] / pn[5]
            dif = abs(gin - rn)

            # Absolute error tolerance satisfied?
            if dif <= acu:
                # Relative error tolerance satisfied?
                if dif <= acu * rn:
                    return 1.0e+00 - factor * gin
            gin = rn

        for i in np.arange(4):
            pn[i] = pn[i + 2]

        if oflo <= np.abs(pn[4]):
            for i in np.arange(4):
                pn[i] = pn[i] / oflo


@njit
@deprecated(reason="Accuracy is not enough")
def gammainc_u(x, s, regularized=True):
    """
    Computes the regularized upper incomplete gamma function
    via modified Lentz's method for computing continued fraction
    https://github.com/compute-io/gammainc/blob/master/lib/number.js
    :param {Number} x - function parameter
    :param {Number} s - function parameter
    :param {Boolean} [regularized=true] - boolean indicating
                if the function should evaluate the regularized or
                non-regularized incomplete gamma functions
    :returns {Number} function value
    """
    if x <= 1.1 or x <= s:
        if regularized:
            return 1.0 - gammainc_l(x, s, regularized)
        else:
            return gamma(s) - gammainc_l(x, s, regularized)

    f = 1.0 + x - s
    C = f
    D = 0
    i = 1

    for i in np.arange(10000):
        a = i * (s - i)
        b = (i << 1) + 1 + x - s
        D = b + a * D
        C = b + a / C
        D = 1 / D
        chg = C * D
        f *= chg
        if np.abs(chg - 1) < EPSILON:
            break

    if regularized:
        return np.exp(s * np.log(x) - x - gammaln(s) - gammaln(f))
    else:
        return np.exp(s * np.log(x) - x - np.log(f))


@njit
@deprecated(reason="Accuracy is not enough")
def gammainc_l(x, s, regularized=True):
    """
    Computes the regularized lower incomplete gamma function
    :param {Number} x - function parameter
    :param {Number} s - function parameter
    :param {Boolean} [regularized=true] - boolean
                   indicating if the function should evaluate the regularized or
                   non-regularized incomplete gamma functions
                   https://github.com/compute-io/gammainc/blob/master/lib/number.js
    :returns {Number} function value
    """
    if x == 0:
        return 0

    if x < 0 or s <= 0:
        return np.nan

    if x > 1.1 and x > s:
        if regularized:
            return 1 - gammainc_u(x, s, regularized)
        else:
            return gamma(s) - gammainc_u(x, s, regularized)

    r = s
    c = 1.
    pws = 1.

    if regularized:
        ft = s * np.log(x) - x - gammaln(s)
    else:
        ft = s * np.log(x) - x

    ft = np.exp(ft)
    while c / pws > EPSILON:
        r += 1
        c *= x / r
        pws += c

    return pws * ft / s


@njit
def gammaln(xvalue):
    """
    ALNGAM computes the logarithm of the gamma function.
    Algorithm AS 245, A Robust and Reliable Algorithm for the Logarithm of the Gamma Function,
    Applied Statistics, Volume 38, Number 2, 1989, pages 397-402.
    https://people.math.sc.edu/Burkardt/f77_src/asa032/asa032.f
    :param xvalue: the argument of the Gamma function.
    :return: real ALNGAM, the logarithm of the gamma function of X.
    """
    alr2pi = 0.918938533204673
    r1 = np.array([-2.66685511495, -24.4387534237, -21.9698958928,
                   11.1667541262, 3.13060547623, 0.607771387771,
                   11.9400905721, 31.4690115749, 15.2346874070])
    r2 = np.array([-78.3359299449, -142.046296688, 137.519416416,
                   78.6994924154, 4.16438922228, 47.0668766060,
                   313.399215894, 263.505074721, 43.3400022514])
    r3 = np.array([-2.12159572323e+05, 2.30661510616e+05, 2.74647644705e+04,
                   -4.02621119975e+04, -2.29660729780e+03, -1.16328495004e+05,
                   -1.46025937511e+05, -2.42357409629e+04, -5.70691009324e+02])

    r4 = np.array([0.279195317918525, 0.4917317610505968, 0.0692910599291889,
                   3.350343815022304, 6.012459259764103])
    xlge = 5.10e+05
    xlgst = 1.0e+30
    x = xvalue
    alngam = 0.0
    # check initial conditions
    if xlgst <= x:
        return np.nan

    if x <= 0.0:
        return np.nan

    # Calculation for 0 < X < 0.5 and 0.5 <= X < 1.5 combined.
    if x < 1.5:
        if x < 0.5:
            alngam = - np.log(x)
            y = x + 1.0

            # Test whether X < machine epsilon
            if y == 1.0:
                return alngam
        else:
            alngam = 0.0
            y = x
            x = (x - 0.5) - 0.5

        alngam = alngam + x * ((((
                                         r1[4] * y
                                         + r1[3]) * y
                                 + r1[2]) * y
                                + r1[1]) * y
                               + r1[0]) / ((((
                                                     y
                                                     + r1[8]) * y
                                             + r1[7]) * y
                                            + r1[6]) * y
                                           + r1[5])

        return alngam

    # Calculation for 1.5 <= X < 4.0.
    if x < 4.0:
        y = (x - 1.0) - 1.0

        alngam = y * ((((
                                r2[4] * x
                                + r2[3]) * x
                        + r2[2]) * x
                       + r2[1]) * x
                      + r2[0]) / ((((
                                            x
                                            + r2[8]) * x
                                    + r2[7]) * x
                                   + r2[6]) * x
                                  + r2[5])

    #  Calculation for 4.0 <= X < 12.0.
    elif x < 12.0:
        alngam = ((((
                            r3[4] * x
                            + r3[3]) * x
                    + r3[2]) * x
                   + r3[1]) * x
                  + r3[0]) / ((((
                                        x
                                        + r3[8]) * x
                                + r3[7]) * x
                               + r3[6]) * x
                              + r3[5])

    # Calculation for 12.0 <= X.
    else:
        y = np.log(x)
        alngam = x * (y - 1.0) - 0.5 * y + alr2pi

        if x <= xlge:
            x1 = 1.0 / x
            x2 = x1 * x1

            alngam = alngam + x1 * ((
                                            r4[2] *
                                            x2 + r4[1]) *
                                    x2 + r4[0]) / ((
                                                           x2 + r4[4]) *
                                                   x2 + r4[3])

    return alngam


@njit
def digamma(x):
    """
    DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
    :param x: the argument of the digamma function.
    :return:digamma, the value of the digamma function at X.
    """
    c = 8.5
    euler_mascheroni = 0.57721566490153286060

    if x <= 0:
        return np.nan
    # Approximation for small argument
    if x <= 0.000001:
        return - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x

    # Reduce to DIGAMA(X + N).
    pval = 0.0
    x2 = x
    while x2 < c:
        pval = pval - 1.0 / x2
        x2 = x2 + 1.0

    # Use Stirling's (actually de Moivre's) expansion
    r = 1.0 / x2
    pval = pval + np.log(x2) - 0.5 * r
    r = r * r

    pval = pval \
              - r * (1.0 / 12.0
                     - r * (1.0 / 120.0
                            - r * (1.0 / 252.0
                                   - r * (1.0 / 240.0
                                          - r * (1.0 / 132.0)))))

    return pval


@njit
def trigamma(x):
    """
    TRIGAMMA calculates trigamma(x) = d^2 log(gamma(x)) / dx^2
    :param x: the argument of the trigamma function.
    :return: trigamma, the value of the trigamma function.
    """
    a = 0.0001
    b = 5.0
    b2 = 0.1666666667
    b4 = -0.03333333333
    b6 = 0.02380952381
    b8 = -0.03333333333

    if x <= 0.0:
        return np.nan

    z = x
    # Use small value approximation if X <= A.
    if x <= a:
        return 1.0 / x / x

    # Increase argument to ( X + I ) >= B.
    pval = 0.0
    while z < b:
        pval = pval + 1.0 / z / z
        z = z + 1.0
    # Apply asymptotic formula if argument is B or greater.
    y = 1.0 / z / z
    pval = pval + 0.5 * y + (1.0
                             + y * (b2
                                            + y * (b4
                                                   + y * (b6
                                                          + y * b8)))) / z

    return pval


@njit
def quastn(prob):
    split1 = 0.425
    split2 = 5
    const1 = 0.180625
    const2 = 1.6
    [A0, A1, A2, A3, A4, A5, A6, A7, B1, B2, B3, B4, B5, B6, B7] = [0.338713287279636661e1,
                                                                    0.133141667891784377e3, 0.197159095030655144e4,
                                                                    0.137316937655094611e5, 0.459219539315498715e5,
                                                                    0.672657709270087009e5, 0.334305755835881281e5,
                                                                    0.250908092873012267e4, 0.423133307016009113e2,
                                                                    0.687187007492057908e3, 0.539419602142475111e4,
                                                                    0.212137943015865959e5, 0.393078958000927106e5,
                                                                    0.287290857357219427e5, 0.522649527885285456e4]

    [C0, C1, C2, C3, C4, C5, C6, C7, D1, D2, D3, D4, D5, D6, D7] = [0.142343711074968358e1,
                                                                    0.463033784615654530e1, 0.576949722146069141e1,
                                                                    0.364784832476320461e1, 0.127045825245236838e1,
                                                                    0.241780725177450612e0, 0.227238449892691846e-1,
                                                                    0.774545014278341408e-3, 0.205319162663775882e1,
                                                                    0.167638483018380385e1, 0.689767334985100005e0,
                                                                    0.148103976427480075e0, 0.151986665636164572e-1,
                                                                    0.547593808499534495e-3, 0.105075007164441684e-8]

    [E0, E1, E2, E3, E4, E5, E6, E7, F1, F2, F3, F4, F5, F6, F7] = [0.665790464350110378e1,
                                                                    0.546378491116411437e1, 0.178482653991729133e1,
                                                                    0.296560571828504891e0, 0.265321895265761230e-1,
                                                                    0.124266094738807844e-2, 0.271155556874348758e-4,
                                                                    0.201033439929228813e-6, 0.599832206555887938e0,
                                                                    0.136929880922735805e0, 0.148753612908506149e-1,
                                                                    0.786869131145613259e-3, 0.184631831751005468e-4,
                                                                    0.142151175831644589e-6, 0.204426310338993979e-14]

    q = prob - 0.5
    if abs(q) > split1:
        r = prob
        if q >= 0:
            r = 1 - prob
        if r <= 0:
            print("F Value Invalid")
            return np.nan
        r = np.sqrt(-np.log(r))
        if r > split2:
            r = r - split2
            pval = ((((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0) /
                    (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + 1))
            if q < 0:
                pval = -pval
            return pval
        else:
            r = r - const2
            pval = ((((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0) /
                    (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + 1))
            if q < 0:
                pval = -pval
            return pval

    else:
        r = const1 - q * q
        pval = q * ((((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0) /
                    (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + 1))
        return pval


@njit
def unif_rand(a=0.0, b=1.0, size=1000):
    """
    This function is used to generate random numbers
    from a uniform distribution (a, b)
    :param a: lower bound
    :param b: higher bound
    :param size: size of output array
    :return: np.array with size
    """
    rv_unif = Uniform(a, b)
    return rv_unif.rand(size)


@njit
def draw_bs_sample(data):
    """
    Randomly resampling repeated elements from the data
    :param data: raw data in a 1D np.array
    :return: a np.array with the same length as data
    """

    size = len(data)
    out_array = np.empty(size)

    idxs = unif_rand(0, size - 1, size=size).astype(np.int_)
    for idx in np.arange(size):
        out_array[idx] = data[idxs[idx]]

    return out_array


@njit
def comb_exact(n, k):
    """The number of combinations of N things taken k at a time.
        This is often expressed as "N choose k".
        Parameters
        ----------
        n : int, ndarray
            Number of things.
        k : int, ndarray
            Number of elements taken.
        Returns
        -------
        val : int, float, ndarray
            The total number of combinations.

    """
    if k > n or n < 0 or k < 0:
        return 0

    m = n + 1
    nterms = min(k, n - k)

    numerator = 1
    denominator = 1
    for j in np.arange(1, nterms + 1):
        numerator *= m - j
        denominator *= j

    return numerator / denominator
