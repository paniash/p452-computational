from library import *

def deltaMatrix0(n: int) -> np.ndarray:
    M = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            if x == y:
                M[x,y] = 1

    return M

def deltaMatrix1(n: int, mu: int) -> np.ndarray:
    M = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            if x+mu == y:
                M[x,y] = 1
            if x+mu >= n:       # to satisfy periodic boundary conditions
                if x+mu-n == y:
                    M[x,y] = 1

    return M

def deltaMatrix2(n: int, mu: int) -> np.ndarray:
    M = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            if x-mu == y:
                M[x,y] = 1
            if x-mu < 0:        # to satisfy periodic boundary conditions
                if x-mu+n == y:
                    M[x,y] = 1

    return M

def matrix(n: int, mu: int = 1, m=0.2):
    d0 = deltaMatrix0(20)
    d1 = deltaMatrix1(20, mu)
    d2 = deltaMatrix2(20, mu)
    A = 1/2 * (d1 + d2 + 2*d0) + m**2 * d0

    return A

#%%
n = 20
A = matrix(n)
matInv = np.zeros((n,n))
for i in range(n):
    b = np.zeros(n)
    b[i] = 1
    x, iterations_j, residue_j = jacobi(A, b, 1e-4)
    matInv[:,i] = x

#%%
A = np.array(read_matrix('a.txt'))
n = len(A)
Ainv = np.zeros((n,n))
for i in range(n):
    b = np.zeros(n)
    b[i] = 1
    x, iterations_j, residue_j = jacobi(A, b, 1e-4)
    Ainv[:,i] = x

#%% Plot
if len(iterations_j) > len(iterations_gs):
    iterations = iterations_j
else:
    iterations = iterations_gs

plt.plot(iterations, residue_j, label="Jacobi")
plt.plot(iterations, residue_gs, label="Gauss-Seidel")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Residue")
