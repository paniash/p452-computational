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
        r"""
        Class describing the output characteristics of a MOSFET device.
        """

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

            self.origvf = self.vForward
            self.origvb = self.vBackward
            self.origif = self.iForward
            self.origib = self.iBackward

        def origplot(self, forORback):
            r"""
            Plots the original dataset present in the file
            `forORback`: string specifying "forward" or "backward"
            """
            if forORback == "forward" or forORback == "Forward":
                plt.plot(self.origvf, self.origif)
                plt.xlabel("$V_{ds}$ (volts)")
                plt.ylabel("$I_{ds}$ (Amps)")
                plt.show()

            elif forORback == "backward" or forORback == "backward":
                plt.plot(self.origvb, self.origib)
                plt.xlabel("$V_{ds}$ (volts)")
                plt.ylabel("$I_{ds}$ (Amps)")
                plt.show()

            else:
                raise NameError("Invalid string!")

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

        def conductivity(self):
            """
            Estimate the conductivity of the device in the linear regime during
            forward sweep of voltage.
            """
            self.condForward = self.slope(self.vForward, self.iForward)
            return self.condForward

        def plotter(self, forORback):
            r"""
            forORback: user-specified string either "forward" or "backward"
            """
            if forORback == "forward" or forORback == "Forward":
                a = fit(self.vForward, self.iForward)[0]
                b = fit(self.vForward, self.iForward)[1]
                x = np.linspace(self.vForward[0], self.vForward[-1], 1000)
                y = a + b * x
                plt.plot(x, y, "g", label="Slope = {} S/m".format(b))
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
                y = a + b * x
                plt.plot(x, y, "g", label="Slope = {} S/m".format(b))
                plt.scatter(self.vBackward, self.iBackward, s=5, label="Experimental")
                plt.legend()
                plt.xlabel("$V_{ds}$ (volts)")
                plt.ylabel("$I_{ds}$ (Amps)")
                plt.title("Output characteristics: Backward sweep")
                plt.show()

            else:
                raise NameError("String value must be either 'forward' or 'backward'!")

    class Transfer:
        r"""
        Class describing the transfer characteristics of a MOSFET device.
        """

        def __init__(self, filename):
            file = read_csv(filename)
            v = [sub[0] for sub in file]
            i = [sub[1] for sub in file]

            self.voltage = list(map(float, v))
            i = list(map(abs, list(map(float, i))))
            self.current = i
            self.rootcurr = list(map(sqrt, i))

        def origplot(self):
            r"""
            Function for the user to see the corresponding plots for the
            original dataset and decide on the linear regimes for the forward
            and backward sweeps in the data.
            """
            plt.plot(self.voltage, self.rootcurr)
            plt.xlabel("$V_g$ (volts)")
            plt.ylabel("$\sqrt{I_{ds}}$ (Amps$^{0.5}$)")
            plt.title("Hysteresis curve obtained for the transfer characteristics")
            plt.show()

        def getLinear(self, llim, ulim):
            r"""
            Only keeps the linear regimes of the curves for both forward and
            backward sweeps in gate bias voltage.

            llim: specifies the lower value for y value threshold
            ulim: specifies the upper value for y value threshold

            The limits `llim` and `ulim` are chosen via the user such that the
            region enclosed within this region for both the forward and
            backward sweeps is linear.
            """
            for index, val in enumerate(self.voltage):
                if self.voltage[index - 1] <= self.voltage[index]:
                    count = index

            # Separate the forward and backward sweep data
            self.vForward = self.voltage[:count]
            self.isqForward = self.rootcurr[:count]

            self.vBackward = self.voltage[count:][::-1]
            self.isqBackward = self.rootcurr[count:][::-1]

            lminval = abs(self.isqForward[0] - llim)
            uminval = abs(self.isqForward[-1] - ulim)
            for index, val in enumerate(self.isqForward):
                if abs(val - llim) < lminval:
                    lminval = abs(val - llim)
                    lIndex = index

                if abs(val - ulim) < uminval:
                    uminval = abs(val - ulim)
                    uIndex = index

            self.vForward = self.vForward[: uIndex + 1][lIndex:]
            self.isqForward = self.isqForward[: uIndex + 1][lIndex:]

            lminval = abs(self.isqBackward[0] - llim)
            uminval = abs(self.isqBackward[-1] - ulim)
            for index, val in enumerate(self.isqBackward):
                if abs(val - llim) < lminval:
                    lminval = abs(val - llim)
                    lIndex = index

                if abs(val - ulim) < uminval:
                    uminval = abs(val - ulim)
                    uIndex = index

            self.vBackward = self.vBackward[: uIndex + 1][lIndex:]
            self.isqBackward = self.isqBackward[: uIndex + 1][lIndex:]

        def mobility(self, length, width, capacitance):
            grad = fit(self.vForward, self.isqForward)[1]
            mu = (2 * length * grad**2) / (width * capacitance)
            return mu

        def vthreshold(self, forORback):
            if forORback == "forward" or forORback == "Forward":
                aForward = fit(self.vForward, self.isqForward)[0]
                bForward = fit(self.vForward, self.isqForward)[1]
                return -aForward / bForward

            elif forORback == "backward" or forORback == "Backward":
                aBackward = fit(self.vBackward, self.isqBackward)[0]
                bBackward = fit(self.vBackward, self.isqBackward)[1]
                return -aBackward / bBackward

            else:
                raise NameError("String should be either 'forward' or 'backward'!")

        def delVthreshold(self):
            return abs(self.vthreshold("forward") - self.vthreshold("backward"))

        def reliability(self):
            r"""
            Finds the reliability factor of the MOSFET device.
            It is defined as the ratio of the maximum channel conductivity
            experimentally achieved in a FET at the maximum gate voltage to the
            maximum channel conductivity expected in a correctly functioning ideal
            FET with the claimed carrier mobility \mu and identical other device
            parameters at the same maximum gate voltage.
            Ref: Nature Materials
            """
            grad = fit(self.vForward, self.isqForward)[1]
            idsMax, idsZero = 0, 0
            vgsMax = 0
            for index, val in enumerate(self.voltage):
                if self.voltage[index] == 60.0:
                    idsMax = self.rootcurr[index]
                    vgsMax = self.voltage[index]

                if self.voltage[index] == 0.0:
                    idsZero = self.rootcurr[index]

            r = ((idsMax - idsZero) / vgsMax) ** 2 / (grad**2)
            return r
