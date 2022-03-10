from library import linear_fit, read_csv
from math import log, e, sqrt

time = [1,15,30,45,60,75,90,105,120,135]
counts = [106,80,98,75,74,73,49,38,37,22]
uncertainty = [10,9,10,9,8,8,7,6,6,5]
# data = read_csv('msfit.txt')

# Average lifetime is given by A = A0 exp(-t/tau)
xvals = time.copy()
yvals = counts.copy()
sigma = uncertainty.copy()
variance = [0 for i in range(len(xvals))]

for i in range(len(xvals)):
    yvals[i] = log(yvals[i],e)
    variance[i] = log(sigma[i]**2,e)

a, b, delA2, delB2, cov, chi2 = linear_fit(xvals, yvals, variance)

dof = len(xvals) - 2
lifetime = -1/b
lifetime_error = lifetime**2 * delB2

t_crit = 1.860  # at 95% significance
N = len(xvals)
n = 5
ybar = 0; avg = 0
for i in range(n):
    ybar += counts[i]/n

for j in range(N):
    avg += counts[i]/N

sigma_full = sqrt(N)
t = (ybar - avg) / (sigma_full/sqrt(n))

print("Lifetime: ", lifetime)
print("Error in lifetime: ", lifetime_error)
print("t = ", t)
print("t_crit = ", t_crit)
