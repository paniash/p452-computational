from library import random_walk, eval_averages
from math import sqrt
import matplotlib.pyplot as plt

N = 200
a = 572
m = 16381
seed = 1029.78493
x, y = random_walk(N, a, m, seed)

# Plotting random walk
plt.plot(x, y)
plt.title("2D Random walk")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("q1_plot.png")

#%%
num_walks = 500
rms = eval_averages(N, a, m, seed, num_walks)[0]

print("RMS distance = {} units".format(rms))
print("sqrt(N) = {} units".format(sqrt(N)))
# COMMENT: Hence the RMS distance and sqrt(N) are approximately equal.

### OUTPUT
# RMS distance = 14.278945029165563 units
# sqrt(N) = 14.142135623730951 units
