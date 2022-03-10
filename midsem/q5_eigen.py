from library import power_method, read_matrix
from math import sqrt, pi
import numpy as np
from numpy import cos, sin

A = read_matrix('mstrimat.txt')
eigval1, eigval2, eigvec1, eigvec2 = power_method(A, np.random.rand(len(A)),
        1e-4, 2)

# Given values of eigenvalues and eigenvectors
b = 2
a, c = -1, -1
n = 5

k = 1
given_eigval1 = b + 2*sqrt(a*c)*cos(k*pi/(n+1))
given_eigvec1 = []
for i in range(5):
    given_eigvec1.append(2*(sqrt(c/a))**k * sin(k*pi*i / (n+1)))

k = 2
given_eigval2 = b + 2*sqrt(a*c)*cos(k*pi/(n+1))
given_eigvec2 = []
for i in range(5):
    given_eigvec2.append(2*(sqrt(c/a))**k * sin(k*pi*1j / (n+1)))

print("Obtained: \n")
print("Eigenvalues: ", eigval1, eigval2)
print("Eigenvectors: ", eigvec1, eigvec2)

print("\nGiven: \n")
print("Eigenvalues: ", given_eigval1, given_eigval2)
print("Eigenvectors: ", given_eigvec1, given_eigvec2)
