from library import read_csv, legendreFit, legendre
import numpy as np
import matplotlib.pyplot as plt

file = read_csv("esem4fit.txt")
xvals = [sub[0] for sub in file]
yvals = [sub[1] for sub in file]

# Convert strings to float
xvals = list(map(float, xvals))
yvals = list(map(float, yvals))

order = 4  # Behaviour of order 4 Legendre polynomial is similar to the variation in data given
params = legendreFit(xvals, yvals, order)
c0, c1, c2, c3, c4 = params

# Plotting the fit in Legendre basis
x = np.linspace(-1, 1, 100)
y = 0
for order, coeff in enumerate(params):
    y += coeff * legendre(x, order)

print(
    "The coefficients in Legendre basis are:\nc0 = {}, c1 = {}, c2 = {}, c3 = {}".format(
        c0, c1, c2, c3
    )
)

plt.scatter(xvals, yvals, s=9, label="Data")
plt.plot(x, y, "g--", label="Fit")
plt.title("Data fit using Legendre basis")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.savefig("q2_plot.png")

### OUTPUT
# The coefficients in Legendre basis are:
# c0 = 0.06965779687186321, c1 = 0.0036240203429268145,
# c2 = -0.012082580199521747, c3 = 0.011426217647052553

## Corresponding plot is `q2_plot.png`
