"""
Author: Ashish Panigrahi

NOTE: NumPy library is used for elementary functions such as inner product, matrix
multiplication, etc.
"""
import numpy as np
import random

"""
Helper Functions
"""
# Read matrix from a file given as a string (space separated file)
def read_matrix(file):
    with open(file, 'r') as f:
        a = [[int(num) for num in line.split(' ')] for line in f]

    return a

# Prints matrix as written on paper
def mat_print(a):
    for i in range(len(a)):
        print(a[i])

# Calculates the norm of a vector x
def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2

    return total**(1/2)

# Vector subtraction
def vec_sub(a, b):
    if (len(a) != len(b)):
        exit()
    else:
        total_vec = [0 for i in range(len(a))]
        for i in range(len(a)):
            total_vec[i] = a[i] - b[i]

    return total_vec

# Matrix multiplication
def matmul(a, b):
    product = [[sum(i*j for i,j in zip(a_row, b_col)) for b_col in zip(*b)] for
            a_row in a]

    return product


# Matrix transpose
def transpose(a):
    tr = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
    return tr

# Inner product
def inner_prod(a, b):
    atr = transpose(a)
    result = matmul(atr, b)
    return result

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
        for j in range(i+1, len(b)):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total)/U[i][i]

    return x

"""
Gauss Jordan
"""



"""
LU Decomposition
"""
def lu_decomposition(A: list, b: list) -> list:
    # Partial pivoting with matrix 'a', vector 'b', and dimension 'n'
    def partial_pivot(A: list, b: list):
        count = 0   # keeps a track of number of exchanges
        n = len(A)
        for i in range(n-1):
            if abs(A[i][i]) == 0:
                for j in range(i+1,n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = A[i], A[j]  # interchange ith and jth rows of matrix 'A'
                        count += 1
                        b[j], b[i] = b[i], b[j]  # interchange ith and jth elements of vector 'b'

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
                    L[i][j] = (A[i][j] - total)/U[j][j]

                else :
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
            total1 += L[i,k]**2

        L[i,i] = np.sqrt(A[i,i] - total1)

        for j in range(i+1, n):
            total2 = 0
            for k in range(i):
                total2 += L[i,k]*L[j,k]

            L[j,i] = 1/L[i,i] * (A[i,j] - total2)

    x = forward_backward(L.T, L, b)
    return x

"""
Jacobi Method
"""
def jacobi(A: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    n = len(A)
    x = np.ones(n)     # define a dummy vector for storing solution vector
    xold = np.zeros(n)
    while np.linalg.norm(xold - x) > tol:
        xold = x.copy()
        for i in range(n):
            total = 0
            for j in range(n):
                if i != j:
                    total += A[i,j] * x[j]

            x[i] = 1/A[i,i] * (b[i] - total)

    return x


"""
Gauss-Seidel
"""
def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    n = len(A)
    x = np.zeros(n)
    k = 0
    x0 = x.copy()

    while True:
        for i in range(n):
            s1, s2 = 0, 0
            for j in range(i):
                s1 += A[i][j]*x[j]
            for j in range(i+1,n):
                s2 += A[i][j]*x0[j]

            x[i] = 1/A[i][i] * (b[i] - s1 - s2)

        if np.linalg.norm(x-x0) < tol:
            return x

        k += 1
        x0 = x.copy()

"""
Conjugate Gradient
"""
def conjgrad(A: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """r
    Function to solve a set of linear equations using conjugate gradient
    method. However, this works strictly for symmetric and positive definite
    matrices only.
    """
    n = len(A)
    x = np.ones(n)
    r = b - A@x
    d = r.copy()
    rprevdot = np.dot(r.T, r)

    for i in range(n):
        Ad = A@d
        alpha = rprevdot / np.dot(d.T, Ad)
        x += alpha*d
        r -= alpha*Ad
        rnextdot = np.dot(r.T, r)

        if np.linalg.norm(r) < tol:
            return x

        else:
            beta = rnextdot / rprevdot
            d = r + beta*d
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
                if (i != j) and (abs(A[i,j]) >= amax):
                    amax = abs(A[i][j])
                    k = i
                    l = j

        return amax, k, l

    def givensRotation(A: np.ndarray, S: np.ndarray, k: int, l: int):
        n = len(A)
        diffA = A[l,l] - A[k,k]
        if abs(A[k][l]) < abs(diffA)*1e-20:
            t = A[k,l]/diffA

        else:
            psi = diffA / (2.0*A[k][l])
            t = 1.0/(abs(psi) + np.sqrt(psi**2 + 1.0))
            if psi < 0.0:
                t = -t

        c = 1.0/np.sqrt(t**2 + 1.0)
        s = t*c
        tau = s/(1.0 + c)
        temp = A[k,l]
        A[k,l] = 0.0
        A[k,k] = A[k,k] - t*temp
        A[l,l] = A[l,l] + t*temp

        for i in range(k):
            temp = A[i,k]
            A[i,k] = temp - s*(A[i,l] + tau*temp)
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])

        for i in range(k+1,l):
            temp = A[k,i]
            A[k,i] = temp - s*(A[i,l] + tau*A[k,i])
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])

        for i in range(l+1,n):
            temp = A[k,i]
            A[k,i] = temp - s*(A[l,i] + tau*temp)
            A[l,i] = A[l,i] + s*(temp - tau*A[l,i])

        for i in range(n):
            temp = S[i,k]
            S[i,k] = temp - s*(S[i,l] + tau*S[i,k])
            S[i,l] = S[i,l] + s*(temp - tau*S[i,l])


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
        total = total/(n-1)
        delAverages.append(total)

    jkAverage = 1/n * sum(delAverages)  # calculate jackknife average

    for j in range(n):
        err = 0
        err += (delAverages[j] - jkAverage)**2

    jkError = err/n     # calculate jackknife standard error

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
        xalphaAvg = 1/n * sum(bootSamples[j])
        xb = 1/B * sum(xalphaAvg)

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
    n = len(xvals)   # number of datapoints

    s, sx, sy, sxx, sxy = 0, 0, 0, 0, 0
    for i in range(n):
        s += 1 / variance[i]**2
        sx += xvals[i] / variance[i]**2
        sy += yvals[i] / variance[i]**2
        sxx += xvals[i]**2 / variance[i]**2
        sxy += xvals[i]*yvals[i] / variance[i]**2

    delta = s*sxx - sx**2
    a = (sxx*sy - sx*sxy) / delta
    b = (s*sxy - sx*sy) / delta

    # calculate chi^2 / dof
    dof = n - 2
    chi2 = 0
    for i in range(n):
        chi2 += (yvals[i] - a - b*xvals[i])**2 / variance[i]**2

    delA2 = sxx / delta; delB2 = s / delta
    cov = -sx / delta
    return a, b, delA2, delB2, cov, chi2


"""
Polynomial Fit
"""
def polynomial(xvals: np.ndarray, yvals: np.ndarray, variance: np.ndarray, k:
        int):
    """r
    xvals, yvals: data points given as a list (separately) as input
    Variance: given as input for every datapoint
    k: the degree of polynomial

    Return values:
        a0, a1, a2
        Quadratic plot: y = a0 + a1*x + a2*x**2
    """
    n = len(xvals)

    sx1 = sum(xvals)
    sx2 = 0
    sx3 = 0
    sx4 = 0

    sy = sum(yvals)
    sxy = 0
    sx2y = 0

    for i in range(n):
        sx2 += xvals[i]**2
        sx3 += xvals[i]**3
        sx4 += xvals[i]**4
        sxy += xvals[i] * yvals[i]
        sx2y += xvals[i]**2 * yvals[i]

    # Construct matrices from the above calculated values
    A = [[n, sx1, sx2], [sx1, sx2, sx3], [sx2, sx3, sx4]]
    b = [sy, sxy, sx2y]

    # Solve for coefficients a0, a1, a2 for, y = a0 + a1*x + a2*x**2
    sol = lin_solver(A, b)

    return sol


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
