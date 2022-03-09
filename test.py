from library import *

A = np.array(read_matrix('a.txt'))
b = np.array([-5/2, 2/3, 3, -4/3, -1/3, 5/3])
# b = np.array([19, 2, 13, -7, -9, 2])

x_lu = lu_decomposition(A, b)
print("LU: ", x_lu)

x_gs = gauss_seidel(A, b, 1e-4)[0]
print("GS: ", x_gs)

# x_conj = conjgrad(A, b, 1e-4)
# print("Conj: ", x_conj)

x_j = jacobi(A, b, 1e-6)[0]
print("Jacobi: ", x_j)

x_gj = gauss_jordan(A, b)
print("Gauss-Jordan: ", x_gj)

# x_cho = cholesky(A, b)
# print("Cholesky: ", x_cho)
