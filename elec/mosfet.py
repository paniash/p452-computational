from elec.read import read_csv
from elec.fit import linear_fit as fit
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def linRegime(voltage: list, current: list, xval: float):
    n = len(voltage)
    maxIter = 0
    min = abs(voltage[0] - xval)
    for i in range(n):
        if abs(voltage[i] - xval) < min:
            min = abs(voltage[i] - xval)
            maxIter = i

    v = voltage[: maxIter + 1]
    i = current[: maxIter + 1]

    return v, i

class Mosfet:
    def __init__(self, output_file, transfer_file):
        self.output = self.Output(output_file)
        self.transfer = self.Transfer(transfer_file)

    class Output:
        def __init__(self, filename):
            self.filename = filename
            file = read_csv(self.filename)
            v = [sub[0] for sub in file]
            i = [sub[1] for sub in file]
            voltage = list(map(abs, list(map(float, v))))
            ids = list(map(abs, list(map(float, i))))

            for i in range(len(voltage)):
                if abs(voltage[i - 1]) <= abs(voltage[i]):
                    count = i

            self.vForward = voltage[:count]
            self.iForward = ids[:count]

            self.vBackward = voltage[count:]
            self.vBackward.reverse()
            self.iBackward = ids[count:]
            self.iBackward.reverse()
            self.condForward, self.condBackward = 0, 0

        def linear(self, valForward, valBackward):
            self.vForward, self.iForward = linRegime(
                self.vForward, self.iForward, valForward
            )
            self.vBackward, self.iBackward = linRegime(
                self.vBackward, self.iBackward, valBackward
            )

        def slope(self, voltage, current):
            a = fit(voltage, current)[1]
            return a

        def hysteresis(self):
            self.condForward = self.slope(self.vForward, self.iForward)
            self.condBackward = self.slope(self.vBackward, self.iBackward)
            return abs(self.condForward - self.condBackward)

        def plotter(self, forORback):
            r"""
            forORback: user-specified string either "forward" or "backward"
            """
            if forORback == "forward" or forORback == "Forward":
                a = fit(self.vForward, self.iForward)[0]
                b = fit(self.vForward, self.iForward)[1]
                x = np.linspace(self.vForward[0], self.vForward[-1], 1000)
                y = a + b*x
                plt.plot(x, y, 'g', label="Slope = {} S/m".format(b))
                plt.scatter(self.vForward, self.iForward, s=5, label="Experimental")
                plt.legend()
                plt.xlabel("$V_{ds}$ (volts)")
                plt.ylabel("$I_{ds}$ (Amps)")
                plt.title("Output characteristics: Forward sweep")
                plt.show()

            elif forORback == "backward" or forORback == "Backward":
                a = fit(self.vBackward, self.iBackward)[0]
                b = fit(self.vBackward, self.iBackward)[1]
                x = np.linspace(self.vBackward[0], self.vBackward[-1], 1000)
                y = a + b*x
                plt.plot(x, y, 'g', label="Slope = {} S/m".format(b))
                plt.scatter(self.vBackward, self.iBackward, s=5, label="Experimental")
                plt.legend()
                plt.xlabel("$V_{ds}$ (volts)")
                plt.ylabel("$I_{ds}$ (Amps)")
                plt.title("Output characteristics: Backward sweep")
                plt.show()


    class Transfer:
        def __init__(self, filename):
            self.filename = filename
            file = read_csv(self.filename)
            v = [sub[0] for sub in file]
            i = [sub[1] for sub in file]

            self.voltage = list(map(float, v))
            i = list(map(abs, list(map(float, i))))
            self.rootcurr = list(map(sqrt, i))

        # Only keeps the linear part of the relevant arrays
        def linear(self):
            for i in range(len(self.voltage)):
                if self.voltage[i-1] < self.voltage[i]:
                    count1 = i

            arr = self.voltage[count1:]
            for i in range(len(arr)):
                if arr[i] < 0:
                    count2 = i
                    break

            self.voltage = arr[:count2]
            self.rootcurr = self.rootcurr[count1:][:count2]

        def mobility(self, length, width, capacitance):
            grad = fit(self.voltage, self.rootcurr)[1]
            mu = 2*length / (width * capacitance) * grad**2
            return mu

        def vthreshold(self):
            a = fit(self.voltage, self.rootcurr)[0]
            b = fit(self.voltage, self.rootcurr)[1]
            return -a/b

        def plotter(self):
            a = fit(self.voltage, self.rootcurr)[0]
            b = fit(self.voltage, self.rootcurr)[1]
            x = np.linspace(self.voltage[0], self.voltage[-1], 1000)
            y = a + b*x

            plt.plot(x, y, 'g', label="Slope = {} S/m".format(b))
            plt.scatter(self.voltage, self.rootcurr, s=5, label="Datapoints")
            plt.legend()
            plt.show()
