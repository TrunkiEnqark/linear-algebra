from matrix import Matrix
import numpy as np

A = Matrix([[1.3, 2.1, 3.5], [3.3, 2.1, 1.2], [2.8, 1.0, 3.4]])
B = [[0, 5, -10],
    [0, 22, 16],
    [0, -9, -2]]

B_mymatrix = Matrix(B)
B_np = np.array(B)
print("My Matrix:")
print("Eigenvalues: ", B_mymatrix.eigenvalues())
print("Eigenvectors: ", B_mymatrix.eigenvectors())

print("Answer:")
print("Eigenvalues: ", np.linalg.eig(B_np)[0])
print("Eigenvectors: ", np.linalg.eig(B_np)[1])
# print(B.inverse())
# print(B.gauss_jordan()[1])