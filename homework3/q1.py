from library import monteCarlo
from math import exp, log

def f(x):
    return exp(-x**2)

def p(x, alpha):
    return alpha * exp(-x)

# Without importance sampling
total_without = monteCarlo(f, 10000)

# With importance sampling
def g(x, alpha=1):
    return f(-log(1 - x/alpha)) / p(x, alpha)

total_with = monteCarlo(g, 10000)

print("Without importance sampling: {}".format(total_without))
print("With importance sampling: {}".format(total_with))

### OUTPUT
# Without importance sampling: 0.7459725214732348
# With importance sampling: 0.7553025885846628
