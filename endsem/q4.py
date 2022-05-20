from math import sqrt

def gaussQuad(func: float, n: int, llim, ulim):
    # Changing interval to general case instead of [-1,1]
    def newf(x, func, llim, ulim):
        return (ulim - llim) / 2 * func((ulim - llim) / 2 * x + (ulim + llim) / 2)

    if n == 4:
        return (
            0.347854845 * newf(0.861136311, func, llim, ulim)
            + 0.652145154 * newf(0.339981043, func, llim, ulim)
            + 0.652145154 * newf(-0.339981043, func, llim, ulim)
            + 0.347854845 * newf(-0.861136311, func, llim, ulim)
        )

    elif n == 5:
        return (
            0.236926885 * newf(0.906179845, func, llim, ulim)
            + 0.478628670 * newf(0.538469310, func, llim, ulim)
            + 0.568888889 * newf(0, func, llim, ulim)
            + 0.478628670 * newf(-0.538469310, func, llim, ulim)
            + 0.236926885 * newf(-0.906179845, func, llim, ulim)
        )

    elif n == 6:
        return (
            0.171324492 * newf(0.932469514, func, llim, ulim)
            + 0.360761573 * newf(0.661209386, func, llim, ulim)
            + 0.467913934 * newf(0.238619186, func, llim, ulim)
            + 0.467913934 * newf(-0.238619186, func, llim, ulim)
            + 0.360761573 * newf(-0.661209386, func, llim, ulim)
            + 0.171324492 * newf(-0.932469514, func, llim, ulim)
        )


# Integrand is obtained using Coulomb's law and is integrated from -L/2 to +L/2
def integrand(x, r=1.0, l=2.0, lamb=1.0):
    return lamb * 1 / (sqrt(r**2 + x**2))


# Evaluating integral
L = 2.0
integral4 = gaussQuad(integrand, 4, -L / 2, L / 2)
integral5 = gaussQuad(integrand, 5, -L / 2, L / 2)
integral6 = gaussQuad(integrand, 6, -L / 2, L / 2)

print("Potential at various degrees of Gaussian Quadrature are")
print("4-point V = {}".format(integral4))
print("5-point V = {}".format(integral5))
print("6-point V = {}".format(integral6))

# Extract digits from a number
import math

def digits(num):
    x = [
        (num // (10**i)) % 10 for i in range(math.ceil(math.log(num, 10)) - 1, -1, -1)
    ]
    return x


# Convert number to string
num4 = "%s" % integral4
num5 = "%s" % integral5
num6 = "%s" % integral6

# Extract decimal portion
dec4 = int(num4.split(".")[1])
dec5 = int(num5.split(".")[1])
dec6 = int(num6.split(".")[1])

# Store individual digits in array
dig4 = digits(dec4)
dig5 = digits(dec5)
dig6 = digits(dec6)

# 9th and 10th decimal places
nine4, ten4 = dig4[8], dig4[9]
nine5, ten5 = dig5[8], dig5[9]
nine6, ten6 = dig6[8], dig6[9]

# Convert to string and concatenate
nine4, ten4 = "%s" % nine4, "%s" % ten4
nine5, ten5 = "%s" % nine5, "%s" % ten5
nine6, ten6 = "%s" % nine6, "%s" % ten6

digits4 = nine4 + ten4
digits5 = nine5 + ten5
digits6 = nine6 + ten6
print("\nThe 9th and 10th decimal places of the obtained integral values are:")
print("4-point : {}".format(digits4))
print("5-point : {}".format(digits5))
print("6-point : {}".format(digits6))

# COMMENT: Hence, the 9th and 10th decimal places are fairly close by for
# 4-point and 6-point values whereas it is not so for 5-point value.


### OUTPUT
# Potential at various degrees of Gaussian Quadrature are
# 4-point V = 1.7620541789046658
# 5-point V = 1.7628552954010728
# 6-point V = 1.7627300484997592
#
# The 9th and 10th decimal places of the obtained integral values are:
# 4-point : 89
# 5-point : 54
# 6-point : 84
