import math
from sympy import symbols, solve

class Matrix:
    def __init__(self, data):
        self.matrix = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
    
    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[self.matrix[i][j] + other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[self.matrix[i][j] - other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    def __mul__(self, other):
        # matrix x matrix
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Columns in first matrix must equal to number of rows in second matrix")
            result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result[i][j] += self.matrix[i][k] * other.matrix[k][j]
            return Matrix(result)
        elif isinstance(other, (int, float)): # matrix x value
            return Matrix([[self.matrix[i][j] * other for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Unsupported operation for __mul__ with type: {}".format(type(other)))
    
    def __truediv__(self, val):
        if isinstance(val, (int, float)):
            return Matrix([[self.matrix[i][j] / val for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Unsupported operation for __truediv__ with type: {}".format(type(val)))

    def transpose(self):
        return Matrix([[self.matrix[i][j] for i in range(self.rows)] for j in range(self.cols)])
    
    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def minor(self, i, j):
        return Matrix([row[:j] + row[j+1:] for row in (self.matrix[:i] + self.matrix[i+1:])])
    
    def determinant(self):
        if self.cols != self.rows:
            raise ValueError("Matrix muse be square")
        if self.rows == 1:
            return self.matrix[0][0]
        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for j in range(self.rows):
            det += (-1)**j * self.matrix[0][j] * self.minor(0, j).determinant()
        return det
    
    def cofactor(self):
        if self.cols != self.rows:
            raise ValueError("Matrix is not invertible")
        
        n = self.rows
        cofactor_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                cofactor_matrix[i][j] = self.minor(i, j).determinant() * (-1)**(i + j)
        return Matrix(cofactor_matrix)
    
    def adjoint(self):
        return self.cofactor().transpose()
    
    def invertible(self):
        return self.determinant() > 1e-12
    
    def inverse(self):
        if not self.invertible():
            raise ValueError("Matrix is not invertible")
        det = self.determinant()
        return self.adjoint() / det

    def gauss_jordan(self): # -> (rank, inverse)
        n = self.rows
        m = self.cols
        original = [row[:] for row in self.matrix]
        if n == m:
            identity = Matrix.identity(n).matrix
        else:
            identity = None
        
        rank = 0
        for col in range(m):
            pivot_row = None
            for row in range(rank, n):
                if abs(original[row][col]) > 1.0e-12:
                    pivot_row = row
                    break
            
            if pivot_row is not None:
                if pivot_row != rank:
                    original[rank], original[pivot_row] = original[pivot_row], original[rank]
                    if identity is not None:
                        identity[rank], identity[pivot_row] = identity[pivot_row], identity[rank]
                
                pivot = original[rank][pivot_row]
                for j in range(m):
                    original[rank][j] /= pivot
                    if identity is not None:
                        identity[rank][j] /= pivot
                
                for i in range(n):
                    if i != rank:
                        factor = original[i][col]
                        for j in range(m):
                            original[i][j] -= factor * original[rank][j]
                            if identity:
                                identity[i][j] -= factor * identity[rank][j]
                
                rank += 1
    
        return rank, Matrix(identity) if identity is not None else None
    
    def rank(self):
        return self.gauss_jordan()[0]   
    
    def eigenvalues(self):
        λ = symbols('λ')
        # det(A - λ*I) = 0
        equation = Matrix([
            [self.matrix[i][j] - (λ if i == j else 0) for j in range(self.cols)]
            for i in range(self.rows)
        ]).determinant()
        
        eigenvalues = solve(equation, λ) # use sympy to solve this equation
        return eigenvalues
    
    def eigenvectors(self):
        pass
    
    def is_positive_definite(self):
        pass
    
    def diagonalize(self):
        pass
    
    def is_positive_definite(self):
        pass
    
    def norm(self):
        pass
    
    def vector_norm(self):
        pass
    
if __name__ == '__main__':
    A = Matrix([[1.3, 2.1, 3.5], [3.3, 2.1, 1.2], [2.8, 1.0, 3.4]])
    B = Matrix([[-5, 2], [-7, 4]])
    print(B.eigenvalues())
    # print(B.inverse())
    # print(B.gauss_jordan()[1])