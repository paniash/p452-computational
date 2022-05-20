from library import random_walk, eval_averages
from math import sqrt
import matplotlib.pyplot as plt

N = 200
a = 572
m = 16381
seed = 1029.78493
x, y = random_walk(N, a, m, seed)

plt.plot(x, y)
plt.title("2D Random walk")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("q1_plot.png")

#%%
num_walks = 500
rms = eval_averages(N, a, m, seed, num_walks)[0]

print("RMS distance = {}".format(rms))
print("sqrt(N) = {}".format(sqrt(N)))
print("Hence RMS distance and sqrt(N) are approximately equal.")

### OUTPUT
# RMS distance = 14.278945029165563
# sqrt(N) = 14.142135623730951
# Hence RMS distance and sqrt(N) are approximately equal.

## Plot file is `q1_plot.png`
