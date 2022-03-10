from library import jacobi, gauss_seidel, read_matrix
import numpy as np

A = read_matrix('msmatinv_A.txt')
b = read_matrix('msmatinv_B.txt')
b = b[0]
tol = 1e-5

x_jacobi = jacobi(A, b, tol)[0]
x_gs = gauss_seidel(A, b, tol)[0]
print("Solutions: \n")
print("Jacobi: ", x_jacobi)
print("Gauss-Seidel: ", x_gs)
