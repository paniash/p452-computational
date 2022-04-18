from elec.read import read_csv
from elec.fit import linear_fit as fit
import numpy as np
import matplotlib.pyplot as plt

file = read_csv("output.csv")
x1 = [sub[0] for sub in file]   # store columns in lists
x2 = [sub[1] for sub in file]
x3 = [sub[2] for sub in file]
x4 = [sub[3] for sub in file]
x5 = [sub[4] for sub in file]
x6 = [sub[5] for sub in file]

# convert string elements into float
voltage = list(map(abs, list(map(float, x1))))
ids1 = list(map(abs, list(map(float, x2))))
ids2 = list(map(abs, list(map(float, x3))))
ids3 = list(map(abs, list(map(float, x4))))
ids4 = list(map(abs, list(map(float, x5))))
ids5 = list(map(abs, list(map(float, x6))))

# Separate into forward and backward data
for i in range(len(voltage)):
    if abs(voltage[i-1]) <= abs(voltage[i]):
        count = i

vForward = voltage[:count]
idsForward = ids3[:count]

vBackward = voltage[count:]
vBackward.reverse()
idsBackward = ids3[count:]
idsBackward.reverse()

# # Function that returns the voltage and current values within the linear region
# def linRegime(voltage: list, current: list, tol: float):
#     def linParams(xvals: np.ndarray, yvals: np.ndarray):
#         n = len(xvals)   # number of datapoints
#         s, sx, sy, sxx, sxy = 0, 0, 0, 0, 0

#         for i in range(n):
#             s += 1
#             sx += xvals[i]
#             sy += yvals[i]
#             sxx += xvals[i]**2
#             sxy += xvals[i]*yvals[i]

#         delta = s*sxx - sx**2
#         a = (sxx*sy - sx*sxy) / delta

#         return a

#     n = len(voltage)
#     slopeOld = linParams(voltage[:5], current[:5])
#     print("SlopeOld", slopeOld)
#     maxIter = 0
#     for i in range(3, n):
#         slope = linParams(voltage[:i], current[:i])
#         print("Slope = ", slope)
#         if abs(slopeOld - slope) < tol:
#             maxIter = i

#     if (voltage[1] - voltage[0]) > 0:
#         v = voltage[:maxIter]
#         i = current[:maxIter]

#     elif (voltage[1] - voltage[0]) < 0:
#         v = voltage[maxIter:]
#         i = current[maxIter:]

#     return v, i

def linRegime(voltage: list, current: list, xval: float):
    n = len(voltage)
    maxIter = 0
    min = abs(voltage[0] - xval)
    for i in range(n):
        if abs(voltage[i] - xval) < min:
            min = abs(voltage[i] - xval)
            maxIter = i

    v = voltage[:maxIter+1]
    i = current[:maxIter+1]

    return v, i

a, b, delA2, delB2, cov, chi2


# class MosFet:
#     def __init__(self, filename):
#         self.filename = filename

#     def voltage(self):
#         file = read_csv(self.filename)
#         v = [sub[0] for sub in file]
#         voltage = list(map(abs, list(map(float, v))))
#         return voltage

#     def iDS(self):
#         file = read_csv(self.filename)
#         i = [sub[1] for sub in file]
#         ids = list(map(abs, list(map(float, i))))
#         return ids

#     def conductivity(self):
#         return voltage(self)
