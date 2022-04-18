from elec.read import read_csv
from elec.fit import linear_fit as fit
import numpy as np
import matplotlib.pyplot as plt

# read file
file = read_csv("p2.txt")
x1 = [sub[0] for sub in file]   # store columns in lists
x2 = [sub[1] for sub in file]
x3 = [sub[2] for sub in file]

# convert string elements into float
vDv = list(map(float, x1))
iD = list(map(abs, list(map(float, x2))))
iS = list(map(abs, list(map(float, x3))))

# Remove negative entries of vDv and corresponding elements from iD and iS
indices_left = []
ileft = 0
for index, v in enumerate(vDv):
    if v < 0:
        indices_left.append(index)
    else:
        break

ileft = max(indices_left)
vDv = vDv[ileft+1:]
iD = iD[ileft+1:]
iS = iS[ileft+1:]

indices_right = []
iright = 0
for index, v in enumerate(vDv[::-1]):
    if v < 0:
        indices_right.append(index)
    else:
        break

iright = max(indices_right)
vDv = vDv[:-iright-1]
iS = iS[:-iright-1]
iD = iD[:-iright-1]

# Separate into forward and backward data
for i in range(len(vDv)):
    if vDv[i-1] <= vDv[i]:
        count = i

vForward = vDv[:count]
iSForward = iS[:count]
iDForward = iD[:count]

vBackward = vDv[count:]
iSBackward = iS[count:]
iDBackward = iD[count:]

# Correct for nonlinear behaviour of backward characteristics
def backCorrection(current: list, voltage: list):
    for i in range(len(current)):
        if current[i-1] >= current[i]:
            maxIndex = i

    current = current[:maxIndex]
    voltage = voltage[:maxIndex]

    return current, voltage

iSBackward, vSBack = backCorrection(iSBackward, vBackward)
iDBackward, vDBack = backCorrection(iDBackward, vBackward)


#%%
a, b, delA2, delB2, cov, chi2 = fit(vDBack, iDBackward)
x = np.linspace(vDBack[0], vDBack[-1], 1000)
y = a + b*x
plt.plot(x, y, label="fit")
plt.scatter(vDBack, iDBackward, s=5, label="data")
plt.legend()
plt.show()

a, b, delA2, delB2, cov, chi2 = fit(vForward, iDForward)
x = np.linspace(vForward[0], vForward[-1], 1000)
y = a + b*x
plt.plot(x, y, label="fit")
plt.scatter(vForward, iDForward, s=5, label="data")
plt.legend()
plt.show()
