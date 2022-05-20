from library import pdeExplicit
from math import sin, pi
import matplotlib.pyplot as plt

#%% u(x, 0) boundary condition at t=0
def g(x: float) -> float:
    return 20 * abs(sin(pi * x))


L = 2.0
nx = 20
tmax = 4.0
nt = 5000
dt = tmax / nt
dx = L / nx

t0 = 0 * dt
t1 = 10 * dt
t2 = 20 * dt
t3 = 50 * dt
t4 = 100 * dt
t5 = 200 * dt
t6 = 500 * dt

# Boundary condition
a = b = 0

u0, x0 = pdeExplicit(L, nx, dt, t0, g, a, b)
u1, x1 = pdeExplicit(L, nx, dt, t1, g, a, b)
u2, x2 = pdeExplicit(L, nx, dt, t2, g, a, b)
u3, x3 = pdeExplicit(L, nx, dt, t3, g, a, b)
u4, x4 = pdeExplicit(L, nx, dt, t4, g, a, b)
u5, x5 = pdeExplicit(L, nx, dt, t5, g, a, b)
u6, x6 = pdeExplicit(L, nx, dt, t6, g, a, b)

#%% Plot
plt.plot(x0, u0, label="t_step = 0")
plt.plot(x1, u1, label="t_step = 10")
plt.plot(x2, u2, label="t_step = 20")
plt.plot(x3, u3, label="t_step = 50")
plt.plot(x4, u4, label="t_step = 100")
plt.plot(x5, u5, label="t_step = 200")
plt.plot(x6, u6, label="t_step = 500")
plt.legend()
plt.xlabel("$x$ units")
plt.ylabel("Temperature ($\degree$C)")
plt.savefig("q3_plot.png")

### DISCUSSION
# The initial temperature gradient follows a sinusoidal curve with the
# antinodes (point where temperature is maximum) being at x=0.5 and x=1.5
# The heat equation governs how this initial temperature gradient varies with
# time. As can be seen from the obtained plot, the temperature is zero at x=0,
# x=2.0 and x=1.0. Given our boundary conditions a(t) and b(t) = 0 at all
# times, the temperature at the extreme points x=0 and x=2 does not change with time.
# However, with time, heat propagates at a finite rate through the rod, thus
# gradually increasing the temperature at x=1, which can be seen in the
# corresponding plot. This temperature gradient disperses with time as seen in
# any diffusion-like phenomena (which follows the same equation) and hence, the
# temperature at x=0.5 and x=1.5 (which were at maximum temperature at t=0)
# reduces as time increases.
