from library import lu_decomposition, jacobi, conjgrad, gauss_seidel
from library import matmul, mat_print
import matplotlib.pyplot as plt

# Define relevant matrices and vectors
A = [[2, -3, 0, 0, 0, 0], [-1, 4, -1, 0, -1, 0], [0, -1, 4, 0, 0, -1], [0, 0,
    0, 2, -3, 0], [0, -1, 0, -1, 4, -1], [0, 0, -1, 0, -1, 4]]
b = [-5/3, 2/3, 3, -4/3, -1/3, 5/3]

# Solution using LU and Jacobi
x_lu = lu_decomposition(A, b)
xj = jacobi(A, b, 1e-4)[0]
xgs = gauss_seidel(A, b, 1e-4)[0]

print("Solutions- \n")
print("LU Decomposition: x = {}".format(x_lu))
print("Jacobi: x = {}".format(xj))
print("Gauss-Seidel: x = {}".format(xgs))

# Calculating inverse of A using Jacobi, Gauss-Seidel and Conjugate Gradient
b1 = [1, 0, 0, 0, 0, 0]
b2 = [0, 1, 0, 0, 0, 0]
b3 = [0, 0, 1, 0, 0, 0]
b4 = [0, 0, 0, 1, 0, 0]
b5 = [0, 0, 0, 0, 1, 0]
b6 = [0, 0, 0, 0, 0, 1]

# Jacobi
jx1, iterj, resigj = jacobi(A, b1, 1e-4)
jx2 = jacobi(A, b2, 1e-4)[0]
jx3 = jacobi(A, b3, 1e-4)[0]
jx4 = jacobi(A, b4, 1e-4)[0]
jx5 = jacobi(A, b5, 1e-4)[0]
jx6 = jacobi(A, b6, 1e-4)[0]

tempMat = [jx1, jx2, jx3, jx4, jx5, jx6]
matInvj = list(zip(*tempMat))   # Inverse of A using Jacobi
print("\nInverse of A using Jacobi method:")
mat_print(matInvj)

# Gauss-Seidel
gsx1, itergs, resigs = gauss_seidel(A, b1, 1e-4)
gsx2 = gauss_seidel(A, b2, 1e-4)[0]
gsx3 = gauss_seidel(A, b3, 1e-4)[0]
gsx4 = gauss_seidel(A, b4, 1e-4)[0]
gsx5 = gauss_seidel(A, b5, 1e-4)[0]
gsx6 = gauss_seidel(A, b6, 1e-4)[0]

tempMat = [gsx1, gsx2, gsx3, gsx4, gsx5, gsx6]
matInvgs = list(zip(*tempMat))  # Inverse of A using Gauss-Seidel
print("\nInverse of A using Gauss-Seidel method:")
mat_print(matInvgs)

# Comparing convergence rates of Jacobi and Gauss-Seidel
plt.plot(iterj, resigj, label="Jacobi")
plt.plot(itergs, resigs, label="Gauss-Seidel")
plt.xlabel("Iterations")
plt.ylabel("Residue")
plt.title("Convergence rate of various methods")
plt.legend()
plt.show()
