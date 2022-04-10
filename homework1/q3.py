from library import conjgrad, kronecker
import matplotlib.pyplot as plt

n = 20
N = n**2

# Indexing approach for the 2D lattice
def aMatrix(a: int, b: int, m = 0.2, tol = 1e-6, n = 20) -> float:
    i, j = a % n, a // n
    k, l = b % n, b // n

    return 0.5 * (kronecker(i+1, a) * kronecker(j, b) + kronecker(i-1, a) *
            kronecker(j, b) - 4 * kronecker(i, a) * kronecker(j, b) +
            kronecker(i, a) * kronecker(j+1, b) + kronecker(i, a) *
            kronecker(j-1, b)) + m**2 * kronecker(i, a) * kronecker(j, b)

matInv = [[0 for i in range(N)] for j in range(N)]
for i in range(N):
    b = [0]*N
    b[i] = 1
    matInv[i] = conjgrad(aMatrix, b, 1e-6)[0]

aInverse = list(zip(*matInv))   # Inverse of A
