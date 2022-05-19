"""
Author: Ashish Panigrahi

NOTE: NumPy library is used for elementary functions such as inner product, matrix
multiplication, etc.
"""
import numpy as np
from math import sqrt
import random

"""
Helper Functions
"""
# Read matrix from a file given as a string (space separated file)
def read_matrix(file):
    with open(file, "r") as f:
        a = [[int(num) for num in line.split(" ")] for line in f]

    return a


# Prints matrix as written on paper
def mat_print(a):
    for i in range(len(a)):
        print(a[i])


# Calculates the norm of a vector x
def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2

    return total ** (1 / 2)


# Vector subtraction
def vec_sub(a, b):
    if len(a) != len(b):
        exit()
    else:
        return [x1 - x2 for (x1, x2) in zip(a, b)]


# Matrix multiplication
def matmul(a, b):
    product = [
        [sum(i * j for i, j in zip(a_row, b_col)) for b_col in zip(*b)] for a_row in a
    ]

    return product


# Matrix and vector multiplication
def vecmul(A, b):
    if len(A) == len(b):
        vec = [0 for i in range(len(b))]
        for i in range(len(b)):
            total = 0
            for j in range(len(b)):
                vec[i] += A[i][j] * b[j]
        return vec


# Matrix transpose
def transpose(a):
    tr = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
    return tr


# Inner product
def inner_prod(a, b):
    atr = transpose(a)
    result = matmul(atr, b)
    return result


# Vector dot product
def dotprod(a, b):
    if len(a) != len(b):
        exit()
    else:
        total = 0
        for i in range(len(a)):
            total += a[i] * b[i]

        return total


def gs_decompose(A):
    U = [[0 for i in range(len(A))] for j in range(len(A))]
    L = [[0 for i in range(len(A))] for j in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A)):
            if i >= j:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]

    return L, U


def read_csv(path):
    with open(path, "r+") as file:
        results = []

        for line in file:
            line = line.rstrip("\n")  # remove `\n` at the end of line
            items = line.split(",")
            results.append(list(items))

        # after for-loop
        return results


##################################################
############ MATRIX INVERSION ####################
##################################################
# Forward-backward substitution function which returns the solution x = [x1, x2, x3, x4]
def forward_backward(U: list, L: list, b: list) -> list:
    y = [0 for i in range(len(b))]

    for i in range(len(b)):
        total = 0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = b[i] - total

    x = [0 for i in range(len(b))]

    for i in reversed(range(len(b))):
        total = 0
        for j in range(i + 1, len(b)):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total) / U[i][i]

    return x


"""
Gauss Jordan
"""


def gauss_jordan(A: list, b: list) -> list:
    def partial_pivot(A: list, b: list):
        n = len(A)
        for i in range(n - 1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i + 1, n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = A[i], A[j]  # interchange A[i] and A[j]
                        b[j], b[i] = b[i], b[j]  # interchange b[i] and b[j]

    n = len(A)
    partial_pivot(A, b)
    for i in range(n):
        pivot = A[i][i]
        b[i] = b[i] / pivot
        for c in range(i, n):
            A[i][c] = A[i][c] / pivot

        for k in range(n):
            if k != i and A[k][i] != 0:
                factor = A[k][i]
                b[k] = b[k] - factor * b[i]
                for j in range(i, n):
                    A[k][j] = A[k][j] - factor * A[i][j]

    x = b
    return x


# def gauss_jordan(A: np.ndarray, b: np.ndarray) -> np.ndarray:
#     # Pivots for a given row k
#     def partial_pivot(A: np.ndarray, b: np.ndarray, k: int) -> tuple:
#         n = len(A)
#         if abs(A[k,k]) < 1e-10:
#             for i in range(k+1, n):
#                 if abs(A[i,k]) > abs(A[k,k]):
#                     A[k], A[i] = A[i], A[k]
#                     b[k], b[i] = b[i], b[k]

#         return A, b

#     n = len(A)
#     for i in range(n):
#         A, b = partial_pivot(A, b, i)
#         # set pivot row
#         pivot = A[i,i]
#         # Divide row with pivot (and corresponding operation on b)
#         for j in range(i, n):
#             A[i,j] /= pivot

#         b[i] /= pivot

#         for j in range(n):
#             if abs(A[j,i]) > 1e-10 and j != i:
#                 temp = A[j,i]
#                 for k in range(i, n):
#                     A[j,k] = A[j,k] - temp * A[i,k]
#                 b[j] = b[j] - temp * b[i]

#     return b


"""
LU Decomposition
"""


def lu_decomposition(A: list, b: list) -> list:
    # Partial pivoting with matrix 'A', vector 'b'
    def partial_pivot(A: list, b: list):
        count = 0  # keeps a track of number of exchanges
        n = len(A)
        for i in range(n - 1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i + 1, n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = (
                            A[i],
                            A[j],
                        )  # interchange ith and jth rows of matrix 'A'
                        count += 1
                        b[j], b[i] = (
                            b[i],
                            b[j],
                        )  # interchange ith and jth elements of vector 'b'

        return A, b, count

    # Crout's method of LU decomposition
    def crout(A: list):
        U = [[0 for i in range(len(A))] for j in range(len(A))]
        L = [[0 for i in range(len(A))] for j in range(len(A))]

        for i in range(len(A)):
            L[i][i] = 1

        for j in range(len(A)):
            for i in range(len(A)):
                total = 0
                for k in range(i):
                    total += L[i][k] * U[k][j]

                if i == j:
                    U[i][j] = A[i][j] - total

                elif i > j:
                    L[i][j] = (A[i][j] - total) / U[j][j]

                else:
                    U[i][j] = A[i][j] - total

        return U, L

    partial_pivot(A, b)
    U, L = crout(A)
    x = forward_backward(U, L, b)
    return x


"""
Cholesky Decomposition
"""


def cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        total1 = 0
        for k in range(i):
            total1 += L[i, k] ** 2

        L[i, i] = np.sqrt(A[i, i] - total1)

        for j in range(i + 1, n):
            total2 = 0
            for k in range(i):
                total2 += L[i, k] * L[j, k]

            L[j, i] = 1 / L[i, i] * (A[i, j] - total2)

    x = forward_backward(L.T, L, b)
    return x


"""
Jacobi Method
"""


def jacobi(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [1 for i in range(n)]  # define a dummy vector for storing solution vector
    xold = [0 for i in range(n)]
    iterations = []
    residue = []
    count = 0
    while norm(vec_sub(xold, x)) > tol:
        iterations.append(count)
        count += 1
        residue.append(norm(vec_sub(xold, x)))
        xold = x.copy()
        for i in range(n):
            total = 0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]

            x[i] = 1 / A[i][i] * (b[i] - total)

    return x, iterations, residue


"""
Gauss-Seidel
"""


def gauss_seidel(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [0 for i in range(n)]
    xold = [1 for i in range(n)]
    iterations = []
    residue = []
    count = 0

    while norm(vec_sub(x, xold)) > tol:
        xold = x.copy()
        iterations.append(count)
        count += 1
        for i in range(n):
            d = b[i]
            for j in range(n):
                if j != i:
                    d -= A[i][j] * x[j]

            x[i] = d / A[i][i]

        residue.append(norm(vec_sub(x, xold)))

    return x, iterations, residue


"""
Conjugate Gradient
"""


def conjgrad(A: list, b: list, tol: float) -> list:
    """r
    Function to solve a set of linear equations using conjugate gradient
    method. However, this works strictly for symmetric and positive definite
    matrices only.
    """
    n = len(b)
    x = [1 for i in range(n)]
    r = vec_sub(b, vecmul(A, x))
    d = r.copy()
    rprevdot = dotprod(r, r)
    iterations = []
    residue = []
    count = 0  # counts the number of iterations

    # convergence in n steps
    for i in range(n):
        iterations.append(count)
        Ad = vecmul(A, d)
        alpha = rprevdot / dotprod(d, Ad)
        for j in range(n):
            x[j] += alpha * d[j]
            r[j] -= alpha * Ad[j]
        rnextdot = dotprod(r, r)
        residue.append(sqrt(rnextdot))
        count += 1

        if sqrt(rnextdot) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            for j in range(n):
                d[j] = r[j] + beta * d[j]
            rprevdot = rnextdot


##################################################
########### EIGENVALUE PROBLEM ###################
##################################################

"""
Power Method
"""


def power_method(A: np.ndarray, x: np.ndarray, tol: float, eignum: int = 1) -> tuple:
    """r
    Function to evaluate the eigenvalues and corresponding eigenvectors for a
    given matrix `A`, a random vector `x` of same dimension, a tolerance `tol`
    and number of eigenvalues `eignum` (either 1 or 2).
    """
    n = len(A)
    x = x / np.linalg.norm(x)
    y = x.copy()
    if eignum == 1:
        diff = 1
        while diff > tol:
            xnew = A @ x
            eigval = np.dot(xnew, x) / np.dot(x, x)
            xnew = xnew / np.linalg.norm(xnew)
            diff = np.linalg.norm(xnew - x)
            x = xnew.copy()

        vec = xnew

        return eigval, vec

    elif eignum == 2:
        diff = 1
        while diff > tol:
            xnew = A @ x
            eigval1 = np.dot(xnew, x) / np.dot(x, x)
            xnew = xnew / np.linalg.norm(xnew)
            diff = np.linalg.norm(xnew - x)
            x = xnew.copy()

        vec1 = xnew

        A = A - eigval1 * np.outer(vec1, vec1.T)
        diff = 1
        while diff > tol:
            ynew = A @ y
            eigval2 = np.dot(ynew, y) / np.dot(y, y)
            ynew = ynew / np.linalg.norm(ynew)
            diff = np.linalg.norm(ynew - y)
            y = ynew.copy()

        vec2 = ynew

        return eigval1, eigval2, vec1, vec2


"""
Jacobi Method (using Given's rotation)
"""


def given_jacobi(A: np.ndarray, tol: float) -> tuple:
    """r
    Generates a transformation matrix to kill the non-zero off-diagonal
    elements and diagonalize the original matrix to find the eigenvalues and
    eigenvectors
    """

    def maxElement(A: np.ndarray) -> tuple:
        """r
        To find the largest off-diagonal element in the matrix
        """
        n = len(A)
        amax = 0.0
        for i in range(n):
            for j in range(n):
                if (i != j) and (abs(A[i, j]) >= amax):
                    amax = abs(A[i][j])
                    k = i
                    l = j

        return amax, k, l

    def givensRotation(A: np.ndarray, S: np.ndarray, k: int, l: int):
        n = len(A)
        diffA = A[l, l] - A[k, k]
        if abs(A[k][l]) < abs(diffA) * 1e-20:
            t = A[k, l] / diffA

        else:
            psi = diffA / (2.0 * A[k][l])
            t = 1.0 / (abs(psi) + np.sqrt(psi**2 + 1.0))
            if psi < 0.0:
                t = -t

        c = 1.0 / np.sqrt(t**2 + 1.0)
        s = t * c
        tau = s / (1.0 + c)
        temp = A[k, l]
        A[k, l] = 0.0
        A[k, k] = A[k, k] - t * temp
        A[l, l] = A[l, l] + t * temp

        for i in range(k):
            temp = A[i, k]
            A[i, k] = temp - s * (A[i, l] + tau * temp)
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])

        for i in range(k + 1, l):
            temp = A[k, i]
            A[k, i] = temp - s * (A[i, l] + tau * A[k, i])
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])

        for i in range(l + 1, n):
            temp = A[k, i]
            A[k, i] = temp - s * (A[l, i] + tau * temp)
            A[l, i] = A[l, i] + s * (temp - tau * A[l, i])

        for i in range(n):
            temp = S[i, k]
            S[i, k] = temp - s * (S[i, l] + tau * S[i, k])
            S[i, l] = S[i, l] + s * (temp - tau * S[i, l])

    n = len(A)
    maxRot = n**2
    S = np.identity(n)
    for i in range(maxRot):
        amax, k, l = maxElement(A)
        if amax < tol:
            return np.diagonal(A), S

        givensRotation(A, S, k, l)


##################################################
####### STATISTICAL DESCRIPTION OF DATA ##########
##################################################
"""
Jackknife: Finds the mean and variance of population via finite sampling
"""


def jackknife(yis: list) -> tuple:
    delAverages = []  # holds all the yk averages for each j
    n = len(yis)
    for i in range(n):
        total = sum(yis)
        total -= yis[i]
        total = total / (n - 1)
        delAverages.append(total)

    jkAverage = 1 / n * sum(delAverages)  # calculate jackknife average

    for j in range(n):
        err = 0
        err += (delAverages[j] - jkAverage) ** 2

    jkError = err / n  # calculate jackknife standard error

    return jkAverage, jkError


"""
Bootstrap: Resampling of data points from unknown distribution
Implementation of empirical bootstrap
"""


def bootstrap(xis: list, B: int) -> tuple:
    bootSamples = []
    n = len(xis)
    for i in range(B):
        xis_sampled = random.choices(xis, k=n)
        bootSamples.append(xis_sampled)

    for j in range(B):
        xalphaAvg = 1 / n * sum(bootSamples[j])
        xb = 1 / B * sum(xalphaAvg)


"""
Linear Regression
"""
# Returns the intercept and slope for a linear regression (chi square fit),
# given a set of data points
def linear_fit(xvals: np.ndarray, yvals: np.ndarray, variance: np.ndarray):
    """r
    xvals, yvals: data points given as a list (separately) as input
    Return values:
        a: intercept
        b: slope
        delA2: variance of a
        delB2: variance of b
        cov: covariance of a, b
        chi2: chi^2 / dof
        Linear plot: y = a + b*x
    """
    n = len(xvals)  # number of datapoints

    s, sx, sy, sxx, sxy = 0, 0, 0, 0, 0
    for i in range(n):
        s += 1 / variance[i] ** 2
        sx += xvals[i] / variance[i] ** 2
        sy += yvals[i] / variance[i] ** 2
        sxx += xvals[i] ** 2 / variance[i] ** 2
        sxy += xvals[i] * yvals[i] / variance[i] ** 2

    delta = s * sxx - sx**2
    a = (sxx * sy - sx * sxy) / delta
    b = (s * sxy - sx * sy) / delta

    # calculate chi^2 / dof
    dof = n - 2
    chi2 = 0
    for i in range(n):
        chi2 += (yvals[i] - a - b * xvals[i]) ** 2 / variance[i] ** 2

    delA2 = sxx / delta
    delB2 = s / delta
    cov = -sx / delta
    return a, b, delA2, delB2, cov, chi2


"""
Polynomial Fit
"""


def polynomial(
    xvals: np.ndarray, yvals: np.ndarray, variance: np.ndarray, degree: int = 1
):
    """r
    xvals, yvals: data points given as a list (separately) as input
    Variance: given as input for every datapoint
    degree: the degree of polynomial

    Return values:
        a0, a1, a2
        Polynomial plot: y = a0 + a1*x + a2*x**2 + ... + a_{n-1}*x^{n-1}

    Returns a linear fit if `degree` not predefined
    """
    n = len(xvals)
    params = degree + 1  # no. of parameters
    A = np.zeros((params, params))  # Matrix
    b = np.zeros(params)  # Vector

    for i in range(params):
        for j in range(params):
            total = 0
            for k in range(n):
                total += xvals[k] ** (i + j) / variance[k] ** 2

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += (xvals[k] ** i * yvals[k]) / variance[k] ** 2

        b[i] = total

    paramsVec = lu_decomposition(A, b)
    C = np.linalg.inv(A)

    # Variance of each parameter
    varcoeff = []
    for i in range(len(C)):
        varcoeff.append(C[i, i])

    # Condition number
    condnum = np.linalg.cond(A)

    return paramsVec, varcoeff, condnum


def chebyshev(x: float, order: int) -> float:
    """
    Modified Chebyshev basis
    """
    if order == 0:
        return 1
    elif order == 1:
        return 2 * x - 1
    elif order == 2:
        return 8 * x**2 - 8 * x + 1
    elif order == 3:
        return 32 * x**3 - 48 * x**2 + 18 * x - 1

def chebyshev_first(x: float, order: int) -> float:
    """
    Chebyshev polynomials of first kind
    """
    if order == 0:
        return 1
    elif order == 1:
        return x
    elif order == 2:
        return 2 * x**2 - 1
    elif order == 3:
        return 4 * x**3 - 3 * x
    elif order == 4:
        return 8*x**4 - 8*x**2 + 1
    elif order == 5:
        return 16 * x**5 - 20 * x**3 + 5*x

def chebyshev_second(x: float, order: int) -> float:
    """
    Chebyshev polynomials of second kind
    """
    if order == 0:
        return 1
    elif order == 1:
        return 2*x
    elif order == 2:
        return 4 * x**2 - 1
    elif order == 3:
        return 8 * x**3 - 4 * x
    elif order == 4:
        return 16 * x**4 - 12 * x**2 + 1
    elif order == 5:
        return 32 * x**5 - 32 * x**3 + 6*x

def chebyfit(xvals: np.array, yvals: np.array, degree: int):
    n = len(xvals)
    params = degree + 1
    A = np.zeros((params, params))
    b = np.zeros(params)

    for i in range(params):
        for j in range(params):
            total = 0
            for k in range(n):
                total += chebyshev(xvals[k], j) * chebyshev(xvals[k], i)

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += chebyshev(xvals[k], i) * yvals[k]

        b[i] = total

    paramsVec = lu_decomposition(A, b)

    # Variance of each parameter
    varcoeff = []
    C = np.linalg.inv(A)
    for i in range(len(C)):
        varcoeff.append(C[i, i])

    # Condition number
    condnum = np.linalg.cond(A)

    return paramsVec, varcoeff, condnum


"""
Discrete Fourier Transform
"""


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n = np.ndarray([i for i in range(N)])
    k = n.T
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)
    return X


"""
Pseudorandom number generator
"""


def mlcg(seed: float, a: float, m: float, num: int) -> list:
    """
    num: Number of random values
    a, m: parameters for the generator
    seed: Starting seed for reproducibility
    """
    x = seed
    rands = []
    for i in range(num):
        x = (a * x) % m
        rands.append(x)

    return rands


def random_walk(N):
    """
    N: Number of steps of the random walk
    """
    pos_x = 0  # instantaneous x coordinate
    pos_y = 0  # instantaneous y coordinate
    x = []  # list of all the x coordinates of the walk
    y = []  # list of all the y coordinates of the walk
    total_randnums = 2 * N
    i = int(N / 2)
    for _ in range(N):
        x.append(pos_x)
        y.append(pos_y)
        theta = 2 * pi * mlcg(173.352, 1103515245, 1, total_randnums)[i]
        dx = cos(theta)
        dy = sin(theta)
        pos_x += dx
        pos_y += dy
        i += 1

    return x, y


# Function to return all the averages, i.e. RMS distance, radial distance and
# average displacement of x and y
def eval_averages(N, num_walks=100):
    """
    N: number of steps of the walk
    num_walks: number of walks to be averaged over for a given N (user-specified)
    """
    rms = 0
    rms_square = 0
    radial_distance = 0
    avg_x = 0
    avg_y = 0
    for i in range(num_walks):
        x, y = random_walk(N)
        x_last = x.pop()  # last x coordinate of the walk
        y_last = y.pop()  # last y coordinate of the walk
        rms_square += x_last**2 + y_last**2
        radial_distance += sqrt(x_last**2 + y_last**2) / float(num_walks)
        avg_x += x_last / float(num_walks)
        avg_y += y_last / float(num_walks)

    rms = sqrt(rms_square / float(num_walks))

    return rms, radial_distance, avg_x, avg_y


"""
Monte Carlo integration
"""


def monteCarlo(func, N):
    # Generate list of N random points between lims
    xrand = mlcg(234.34, 65, 1, N)

    summation = 0
    for i in range(N):
        summation += func(xrand[i])

    total = 1 / float(N) * summation

    return total


"""
RK4 for coupled ODEs
"""


def forward_euler(dydx, y0, x0, xf, step_size):
    """Yields solution from x=x0 to x=xf"""
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    n = int((xf - x0) / step_size)  # no. of steps
    for i in range(n):
        x.append(x[i] + step_size)

    for i in range(n):
        y.append(y[i] + step_size * dydx(y[i], x[i]))

    return x, y


def rk4coupled(d2ydx2, dydx, x0, y0, z0, xf, step_size):
    """
    Yields solution from x=x0 to x=xf
    y(x0) = y0 & y'(x0) = z0
    z = dy/dx
    NOTE: y is a vector (hence z is a vector), x is a scalar
    """
    size = len(y0)  # no. of components of y vector
    n = int((xf - x0) / step_size)  # no. of steps
    y = [0] * size
    z = [0] * size

    for comp in range(size):
        x = []
        y[comp] = []
        z[comp] = []

        x.append(x0)
        y[comp].append(y0[comp])
        z[comp].append(z0[comp])

        for i in range(n):
            x.append(x[i] + step_size)
            k1 = step_size * dydx(x[i], y[comp][i], z[comp][i])
            l1 = step_size * d2ydx2(x[i], y[comp][i], z[comp][i])
            k2 = step_size * dydx(
                x[i] + step_size / 2, y[comp][i] + k1 / 2, z[comp][i] + l1 / 2
            )
            l2 = step_size * d2ydx2(
                x[i] + step_size / 2, y[comp][i] + k1 / 2, z[comp][i] + l1 / 2
            )
            k3 = step_size * dydx(
                x[i] + step_size / 2, y[comp][i] + k2 / 2, z[comp][i] + l2 / 2
            )
            l3 = step_size * d2ydx2(
                x[i] + step_size / 2, y[comp][i] + k2 / 2, z[comp][i] + l2 / 2
            )
            k4 = step_size * dydx(x[i] + step_size, y[comp][i] + k3, z[comp][i] + l3)
            l4 = step_size * d2ydx2(x[i] + step_size, y[comp][i] + k3, z[comp][i] + l3)

            y[comp].append(y[comp][i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            z[comp].append(z[comp][i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6)

    return x, y, z


def runge_kutta(d2ydx2, dydx, x0, y0, z0, xf, step_size):
    x = []
    y = []
    z = []  # dy/dx
    x.append(x0)
    y.append(y0)
    z.append(z0)

    n = int((xf - x0) / step_size)  # no. of steps
    for i in range(n):
        x.append(x[i] + step_size)
        k1 = step_size * dydx(x[i], y[i], z[i])
        l1 = step_size * d2ydx2(x[i], y[i], z[i])
        k2 = step_size * dydx(x[i] + step_size / 2, y[i] + k1 / 2, z[i] + l1 / 2)
        l2 = step_size * d2ydx2(x[i] + step_size / 2, y[i] + k1 / 2, z[i] + l1 / 2)
        k3 = step_size * dydx(x[i] + step_size / 2, y[i] + k2 / 2, z[i] + l2 / 2)
        l3 = step_size * d2ydx2(x[i] + step_size / 2, y[i] + k2 / 2, z[i] + l2 / 2)
        k4 = step_size * dydx(x[i] + step_size, y[i] + k3, z[i] + l3)
        l4 = step_size * d2ydx2(x[i] + step_size, y[i] + k3, z[i] + l3)

        y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        z.append(z[i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6)

    return x, y, z


# Returns the result of Lagrange's interpolation formula
def lagrange_interpolation(zeta_h, zeta_l, yh, yl, y):
    zeta = zeta_l + (zeta_h - zeta_l) * (y - yl) / (yh - yl)
    return zeta


# Solves 2nd order ODE given Dirichlet boundary conditions
def shooting_method(
    d2ydx2, dydx, x0, y0, xf, yf, z_guess1, z_guess2, step_size, tol=1e-6
):
    """x0: Lower boundary value of x
    y0 = y(x0)
    xf: Upper boundary value of x
    yf = y(xf)
    z = dy/dx
    """
    x, y, z = runge_kutta(d2ydx2, dydx, x0, y0, z_guess1, xf, step_size)
    yn = y[-1]

    if abs(yn - yf) > tol:
        if yn < yf:
            zeta_l = z_guess1
            yl = yn

            x, y, z = runge_kutta(d2ydx2, dydx, x0, y0, z_guess2, xf, step_size)
            yn = y[-1]

            if yn > yf:
                zeta_h = z_guess2
                yh = yn

                # calculate zeta using Lagrange interpolation
                zeta = lagrange_interpolation(zeta_h, zeta_l, yh, yl, yf)

                # using this zeta to solve using RK4
                x, y, z = runge_kutta(d2ydx2, dydx, x0, y0, zeta, xf, step_size)
                return x, y, z

            else:
                print("Bracketing FAIL! Try another set of guesses.")

        elif yn > yf:
            zeta_h = z_guess1
            yh = yn

            x, y, z = runge_kutta(d2ydx2, dydx, x0, y0, z_guess2, xf, step_size)
            yn = y[-1]

            if yn < yf:
                zeta_l = z_guess2
                yl = yn

                # calculate zeta using Lagrange interpolation
                zeta = lagrange_interpolation(zeta_h, zeta_l, yh, yl, yf)

                x, y, z = runge_kutta(d2ydx2, dydx, x0, y0, zeta, xf, step_size)
                return x, y, z

            else:
                print("Bracketing FAIL! Try another set of guesses.")

    else:
        return x, y, z  # bang-on solution with z_guess1


"""
Gaussian Quadrature
"""


def gaussQuad(func, n, llim, ulim):
    # Change of variable for converting to interval [-1,1]
    def newf(x, func, llim, ulim):
        return (ulim - llim) / 2 * func((ulim - llim) / 2 * x + (ulim + llim) / 2)

    if n == 1:
        return 2 * newf(0, func, llim, ulim)

    elif n == 2:
        return newf(sqrt(1 / 3), func, llim, ulim) + newf(
            -sqrt(1 / 3), func, llim, ulim
        )

    elif n == 3:
        return (
            8 / 9 * newf(0, func, llim, ulim)
            + 5 / 9 * newf(sqrt(3 / 5), func, llim, ulim)
            + 5 / 9 * newf(-sqrt(3 / 5), func, llim, ulim)
        )

    elif n == 4:
        return (
            (18 + sqrt(30))
            / 36
            * newf(sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5)), func, llim, ulim)
            + (18 + sqrt(30))
            / 36
            * newf(-sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5)), func, llim, ulim)
            + (18 - sqrt(30))
            / 36
            * newf(sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5)), func, llim, ulim)
            + (18 - sqrt(30))
            / 36
            * newf(-sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5)), func, llim, ulim)
        )

    elif n == 5:
        return (
            128 / 225 * newf(0, func, llim, ulim)
            + (322 + 13 * sqrt(70))
            / 900
            * newf(1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)), func, llim, ulim)
            + (322 + 13 * sqrt(70))
            / 900
            * newf(-1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)), func, llim, ulim)
            + (322 - 13 * sqrt(70))
            / 900
            * newf(-1 / 3 * sqrt(5 + 2 * sqrt(10 / 7)), func, llim, ulim)
            + (322 - 13 * sqrt(70))
            / 900
            * newf(1 / 3 * sqrt(5 + 2 * sqrt(10 / 7)), func, llim, ulim)
        )


"""
Partial Differential Equations: Diffusion equation
"""


def k(x, mu=0.5, sigma=0.05):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def g(x, L=1.0):
    return np.sin(np.pi * x / L)


def pdeExplicit(L: int, n: int, dt: float, tmax: float, g: float, a=0, b=0):
    """
    L: [0,L] spatial domain
    n: number of spatial mesh points
    tmax: propagation time
    dt: time step
    g: g(x) boundary condition at t=0 points
    a, b: a(t), b(t) boundary conditions at x=0 / x=L points
    """
    # n+1 is the number of mesh points
    u = [0] * (n + 1)
    unew = [0] * (n + 1)

    # Since a(t) = b(t) = 0
    u[0], unew[0] = a, a
    u[n], unew[n] = b, b

    # Step size
    dx = L / n
    alpha = dt / (dx**2)

    xvals = [0]
    for i in range(1, n):
        x = i * dx
        u[i] = g(x)
        xvals.append(x)

    xvals.append(xvals[-1] + dx)

    # Time iteration
    t = dt
    while t < tmax:
        for i in range(1, n):
            # differential eqn
            unew[i] = alpha * u[i - 1] + (1 - 2 * alpha) * u[i] + alpha * u[i + 1]
            u[i] = unew[i].copy()
        t += dt

    return u, xvals


def pdeImplicit(L: int, n: int, dt: float, tmax: float, g: float, a=0, b=0):
    """
    L: [0,L] spatial domain
    n: number of spatial mesh points
    tmax: propagation time
    dt: time step
    g: g(x) boundary condition at t=0 points
    a, b: a(t), b(t) boundary conditions at x=0 / x=L points
    """
    dx = L / (n + 1)
    alpha = dt / (dx**2)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1 + 2 * alpha
        for j in range(n):
            if j == i + 1 or j == i - 1:
                A[i, j] = -alpha

    x = [0]
    for i in range(n - 1):
        x.append(x[i] + dx)

    # At time t=0
    v0 = []
    for i in range(n):
        v0.append(g(x[i], L))
    v0[-1] = v0[0]

    v0 = np.array(v0)
    v = v0.copy()
    for _ in range(int(tmax / dt)):
        v = np.matmul(np.linalg.inv(A), v)

    return v, x
