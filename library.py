"""
NumPy library is used for elementary functions such as inner product, matrix
multiplication, etc.
"""
import numpy as np

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

"""
Gauss Jordan
"""
# def gauss_jordan(A, b):
#     for row in range(len(A)):




"""
LU Decomposition
"""
def lu_decomposition(A: list, b: list) -> list:
    # Partial pivoting with matrix 'a', vector 'b', and dimension 'n'
    def partial_pivot(A: list, b: list, n: int):
        count = 0   # keeps a track of number of exchanges
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

    # Forward-backward substitution function which returns the solution x = [x1, x2, x3, x4]
    def forward_backward(U, L, b):
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

    partial_pivot(A, b, len(A))
    U, L = crout(A)
    x = forward_backward(U, L, b)
    return x

"""
Gauss-Seidel
"""
def gauss_seidel(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [0 for i in range(n)]
    k = 0
    x0 = [0 for i in range(n)]

    while True:
        for i in range(n):
            s1, s2 = 0, 0
            for j in range(i):
                s1 += A[i][j]*x[j]
            for j in range(i+1,n):
                s2 += A[i][j]*x0[j]

            x[i] = 1/A[i][i] * (b[i] - s1 - s2)

        if norm(vec_sub(x, x0)) < tol:
            return x

        k += 1
        x0 = x.copy()

"""
Conjugate Gradient
"""
def conjgrad(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = np.ones(n)
    r = b - np.matmul(A, x)
    d = r.copy()
    rprevdot = np.dot(np.transpose(r), r)

    for i in range(n):
        Ad = np.matmul(A,d)
        alpha = rprevdot / np.dot(np.transpose(d), Ad)
        x += alpha*d
        r -= alpha*Ad
        rnextdot = np.dot(np.transpose(r), r)
        if norm(r) < tol:
            return x
        else:
            beta = rnextdot / rprevdot
            d = r + beta*d
            rprevdot = rnextdot

"""
Power Method
"""
# def power_method(A: np.ndarray, x: np.ndarray, tol: float):
#     n = len(A)
#     lamb = -1.0; lambold = 0.0
#     v = x.copy(); vold = np.zeros(n)
#     while True:
#         vold = v.copy(); v = np.matmul(A, v)
#         lambold = lamb; lamb = (np.dot(v, x)) / (np.dot(vold, x))

#         if (np.abs(lamb - lambold)) < tol:
#             return lamb, v

# def power_method(A: np.ndarray, x: np.ndarray, tol: float):
#     n = len(A)
#     xold = x.copy(); y=x.copy()
#     lambold = 0.0
#     while True:
#         xold = x.copy()
#         x = np.matmul(A, x)
#         lamb = np.dot(x, y) / np.dot(xold, y)
#         print(lamb)

def power_method(A: np.ndarray, x: np.ndarray, k: int, eignum: int = 1) -> tuple:
    lamb1 = 0.0;
    xold = x.copy();
    if eignum == 1:
        for i in range(k):
            x = A @ x
            lamb1 = np.max(x)
            x = x/lamb1

            return lamb1, x

    elif eignum == 2:
        lambs = []; xs = []
        lambs.append(lamb1)
        xs.append(x)

        # Define reduced matrix
        U = x/np.linalg.norm(x)
        Ap = A - lamb1 * U @ U.transpose()

        lamb2 = 0.0;
        for i in range(k):
            xold = Ap @ xold
            lamb2 = np.max(xold)
            xold = xold/lamb2

        lambs.append(lamb2)
        xs.append(xold)
        return lambs, xs


"""
Jacobi Method (using Given's rotation)
"""
def jacobi(A: np.ndarray, tol: float) -> tuple:
    def maxElement(A: np.ndarray) -> tuple:   # Find largest off-diagonal element A[k][l]
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
