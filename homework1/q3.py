import numpy as np
import matplotlib.pyplot as plt


def kronecker(i, j):
    return 1 if i == j else 0

# Indexing approach for the 2D lattice
def aMatrix(x: int, y: int, m=0.2, dim=20) -> float:
    i, j = x // dim, x % dim
    a, b = y // dim, y % dim

    term = 0.5 * (
        (kronecker(i + 1, a) * kronecker(j, b))
        + (kronecker(i - 1, a) * kronecker(j, b))
        - (4 * kronecker(i, a) * kronecker(j, b))
        + (kronecker(i, a) * kronecker(j + 1, b))
        + (kronecker(i, a) * kronecker(j - 1, b))
    ) + (m**2) * kronecker(i, a) * kronecker(j, b)
    return term


def func_multiply(func, x):
    n = len(x)
    prod = np.zeros(n)
    for i in range(n):
        prod[i] = sum([func(i, j) * x[j] for j in range(n)])

    return prod


def conjgrad_onfly(func, b, tol):
    n = len(b)
    count = 0
    x = np.zeros(n)
    r = b - func_multiply(func, x)
    d = np.copy(r)
    residue = [np.linalg.norm(r)]
    iterations = [count]
    for i in range(n):
        Ad = func_multiply(func, d)
        rprevdot = np.dot(r, r)
        alpha = rprevdot / np.dot(d, Ad)
        x += alpha * d
        r -= alpha * Ad
        rnextdot = np.dot(r, r)
        count += 1
        iterations.append(count)
        residue.append(np.linalg.norm(r))

        if np.linalg.norm(r) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            d = r + beta * d
            rprevdot = rnextdot


def aInverse(A, tol, N=400):
    inv = []
    B = np.identity(N)
    for i in range(3):
        x, iter, residue = conjgrad_onfly(A, B[:, i], tol)
        inv.append(x)

    return np.array(inv).T, iter, residue


x, iterations, residue = aInverse(aMatrix, 1e-6)
print("Matrix inverse: ")
print(x)

# Plot of convergence rate
plt.plot(iterations, residue)
plt.xlabel("Iterations")
plt.ylabel("Residue")
plt.show()
