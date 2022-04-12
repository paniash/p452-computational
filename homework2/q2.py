from library import monteCarlo, mlcg
from math import sqrt

def integrand(x):
    return sqrt(1-x**2)

# Assume circle with unit radius
def circle(x, y):
    return x**2 + y**2 - 1

def piEstimate(N):
    xrand = mlcg(34.56, 65, 1, N)
    yrand = mlcg(63.21, 23, 1, N)

    hits = 0
    for i in range(N):
        if circle(xrand[i], yrand[i]) <= 0:
            hits += 1

    estim = hits / N
    return estim

integrated_pi = monteCarlo(integrand, 100000)
hit_pi = piEstimate(10000)
print("Estimated value of pi using integration = {}".format(4*integrated_pi))
print("Estimated value of pi using hits = {}".format(4*hit_pi))
