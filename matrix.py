tol = 1.0e-12 # tolerance
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
    
    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.matrix[i][j] - other.matrix[i][j]) > tol:
                    return False
        return True

    def transpose(self):
        return Matrix([[self.matrix[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def is_symmetric(self):
        return self == self.transpose()
    
    def is_square(self):
        return self.cols == self.rows
    
    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def minor(self, i, j):
        return Matrix([row[:j] + row[j+1:] for row in (self.matrix[:i] + self.matrix[i+1:])])
    
    def determinant(self):
        if not self.is_square():
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
        if not self.is_square():
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
        if self.is_square():
            identity = Matrix.identity(n).matrix
        else:
            identity = None
        
        rank = 0
        for col in range(m):
            pivot_row = None
            for row in range(rank, n):
                if abs(original[row][col]) > tol:
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
    
    def qr_decomposition(self):
        n = self.rows
        m = self.cols
        
        Q = [[0.0] * m for _ in range(n)]
        R = [[0.0] * m for _ in range(m)]
        
        for j in range(m):
            v = [self.matrix[i][j] for i in range(n)]
            
            for i in range(j):
                q_i = [Q[k][i] for k in range(n)]
                R[i][j] = sum(x * y for x, y in zip(q_i, v))
                # v = vector_sub(v, scalar_multiply(q_i, R[i][j]))
                v = [v[p] - q_i[i] * R[i][j] for p in range(n)]
            
            R[j][j] = Matrix.vector_norm(v)
            q_j = Matrix.normalize(v)
            
            for i in range(n):
                Q[i][j] = q_j[i]
        
        return Matrix(Q), Matrix(R)
    
    def eigenvalues(self, max_iters=100):
        if not self.is_square():
            raise ValueError("Matrix must be square")
        
        matrix_k = Matrix(self.matrix)
        
        for _ in range(max_iters):
            Q, R = matrix_k.qr_decomposition()
            matrix_next = R * Q
            
            # Check for convergence
            off_diagonal_sum = sum(
                abs(matrix_next.matrix[i][j]) 
                for i in range(self.rows) 
                for j in range(i)
            )
            if off_diagonal_sum < tol:
                break
            
            matrix_k = matrix_next
        
        return [matrix_k.matrix[i][i] for i in range(self.rows)]

    
    def solve_homogeneous_system(self):
        n = self.rows
        augmented = [row[:] for row in self.matrix]
        
        # Gaussian elimination
        for i in range(n):
            max_element = abs(augmented[i][i])
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > max_element:
                    max_element = abs(augmented[k][i])
                    max_row = k
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            if augmented[i][i] < tol:
                continue
            
            for k in range(i + 1, n):
                c = -augmented[k][i] / augmented[i][i]
                for j in range(i, n):
                    if i == j:
                        augmented[k][j] = 0
                    else:
                        augmented[k][j] += c * augmented[i][j]
        
        # Back substitution
        x = [0] * n
        x[n - 1] = 1  # Assuming non-zero solution exists
        for i in range(n - 2, -1, -1):  
            s = sum(augmented[i][j] * x[j] for j in range(i + 1, n))
            x[i] = -s / augmented[i][i] if abs(augmented[i][i]) > tol else 1
        
        return x
    
    # find eigenvector for corresponding eigenvalue
    def eigenvector(self, eigenvalue):
        if not self.is_square():
            raise ValueError("Matrix must be square")
        
        A_minus_lambda_I = Matrix([
            [self.matrix[i][j] - (eigenvalue if i == j else 0) for j in range(self.cols)] 
            for i in range(self.rows)
        ])
        
        # Solve A_minus_lambda_I * x = 0
        homogeneous_solution = A_minus_lambda_I.solve_homogeneous_system()
        
        # Check if solution is valid
        if all(abs(x) < tol for x in homogeneous_solution):
            return homogeneous_solution
        
        return Matrix.normalize(homogeneous_solution)
    
    def eigenvectors(self):
        if not self.is_square():
            raise ValueError("Matrix must be square")
        
        eigenvalues = self.eigenvalues()
        eigenvectors = []
        
        for eigenvalue in eigenvalues:
            A_minus_lambda_I = Matrix([
                [self.matrix[i][j] - (eigenvalue if i == j else 0) for j in range(self.cols)] 
                for i in range(self.rows)
            ])
            
            # Solve A_minus_lambda_I * x = 0
            homogeneous_solution = A_minus_lambda_I.solve_homogeneous_system()
            
            # Check if solution is valid
            if all(abs(x) < tol for x in homogeneous_solution):
                continue
            
            eigenvectors.append(Matrix.normalize(homogeneous_solution))
        
        return eigenvectors

    def is_positive_definite(self):
        if not self.is_square():
            raise ValueError("Matrix must be square")
        if not self.is_symmetric():
            raise ValueError("Matrix must be symmetric")
        
        return all(e > tol for e in self.eigenvalues())
    
    def diagonalize(self):
        if not self.is_square():
            raise ValueError("Matrix must be square")

        eigenvalues = self.eigenvalues()
        eigenvectors = self.eigenvectors()

        if len(eigenvectors) != self.rows:
            raise ValueError("Matrix is not diagonalizable")

        P = Matrix([[vec[i] for vec in eigenvectors] for i in range(self.rows)])
        D = Matrix([[eigenvalues[i] if i == j else 0 for j in range(self.cols)] for i in range(self.rows)])
        # P*D*P^(-1)
        result = P * D * P.inverse()
        
        if all(abs(result.matrix[i][j] - self.matrix[i][j]) < tol for i in range(self.rows) for j in range(self.cols)):
            return P, D
        else:
            raise ValueError("Matrix is not diagonalizable")
    
    # using Frobenius Norm
    # https://mathworld.wolfram.com/FrobeniusNorm.html
    def norm(self):
        return sum(self.matrix[i][j]**2 for i in range(self.rows) for j in range(self.cols)) ** 0.5
    
    # using L^2 norm
    # https://mathworld.wolfram.com/L2-Norm.html
    @staticmethod
    def vector_norm(vector):
        return sum(x**2 for x in vector) ** 0.5

    @staticmethod
    def normalize(vector):
        norm = Matrix.vector_norm(vector)
        return [x / norm for x in vector] if norm > tol else vector
    
