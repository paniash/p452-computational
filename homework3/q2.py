from library import shooting_method
from math import pi
import numpy as np
import matplotlib.pyplot as plt

#%%
# Eigenenergies for n=1 and n=2 states
E1 = pi**2
E2 = 4 * pi**2

def dzdx1(x, y, z):
    return -E1 * y

def dzdx2(x, y, z):
    return -E2 * y

def dydx(x, y, z):
    return z

x0 = 0.0
xf = 1.0

# Dirichlet boundary conditions
y0 = 0.0
yf = 0.0

z1_guess1 = 3.0
z1_guess2 = 0.0

z2_guess1 = -1.0
z2_guess2 = 3.0
dx = 0.02

x1, y1, z1 = shooting_method(dzdx1, dydx, x0, y0, xf, yf, z1_guess1, z1_guess2, dx)
x2, y2, z2 = shooting_method(dzdx2, dydx, x0, y0, xf, yf, z2_guess1, z2_guess2, dx)

# Normalizing the obtained wavefunctions
y1 = np.array(y1) / np.linalg.norm(y1)
y2 = np.array(y2) / np.linalg.norm(y2)

#%% Plotting
plt.plot(x1, y1, label="n=1")
plt.plot(x2, y2, label="n=2")
plt.xlabel("$x$ coordinate")
plt.ylabel("$\psi$")
plt.legend()
plt.title("Infinite well potential solution")
plt.savefig("q2_plot.png")
