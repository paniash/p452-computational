from library import read_csv, lu_decomposition
import numpy as np
import matplotlib.pyplot as plt

def polyLeastSquare(xvals: np.array, yvals: np.array, degree: int = 1):
    """r
    xvals, yvals: data points given as a list (separately) as input
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
                total += xvals[k] ** (i + j)

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += xvals[k] ** i * yvals[k]

        b[i] = total

    paramsVec = lu_decomposition(A, b)
    return paramsVec


file = read_csv("assign2fit.txt")
xvals = [sub[0] for sub in file]
yvals = [sub[1] for sub in file]

xvals = list(map(float, xvals))
yvals = list(map(float, yvals))

params = polyLeastSquare(xvals, yvals, 3)
# Condition number of matrix = 21980.9
a0, a1, a2, a3 = params[0], params[1], params[2], params[3]

x = np.linspace(0, 1, 100)
y = a0 + a1 * x + a2 * x**2 + a3 * x**3
plt.scatter(xvals, yvals, s=5, label="Datapoints")
plt.plot(x, y, "r", label="Line fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Cubic-fit polynomial")
plt.legend()
plt.show()

# Modified Chebyshev basis
# We solve for the coefficients using LU decomposition
A = [[1, -1, 1, -1], [0, 2, -8, 18], [0, 0, 8, -48], [0, 0, 0, 32]]
b = [a0, a1, a2, a3]
chebyshevs = lu_decomposition(A, b)
chebycoef0, chebycoef1, chebycoef2, chebycoef3 = (
    chebyshevs[0],
    chebyshevs[1],
    chebyshevs[2],
    chebyshevs[3],
)
