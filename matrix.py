import math

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
    
    # A(mxn) x B(nxp)
    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Columns in first matrix must equal to number of rows in second matrix")
        # mxp
        result = [[0 for _ in range(self.rows)] for _ in range(other.cols)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return Matrix(result)                    
    
    def transpose(self):
        return Matrix([[self.matrix[i][j] for i in range(self.rows)] for j in range(self.cols)])
    
    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def determinant(self):
        if self.cols != self.rows:
            raise ValueError("Matrix muse be square")
        if self.rows == 1:
            return self.matrix[0][0]
        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for j in range(self.rows):
            submatrix = Matrix([row[:j] + row[j+1:] for row in self.matrix[1:]])
            det += (-1)**j * self.matrix[0][j] * submatrix.determinant()
        return det
    
if __name__ == '__main__':
    A = Matrix([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    B = Matrix([[3, 4, 5], [9, 8, 3], [6, 7, 7]])
    print(A.determinant())