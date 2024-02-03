import numpy as np

def build_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = []
    for line in lines:
        a, b, x, y = map(float, line.split())

        row1 = [0, 0, 0, x, y, 1, -b*x, -b*y, -b]
        data.append(row1)

        row2 = [x, y, 1, 0, 0, 0, -a*x, -a*y, -a]
        data.append(row2)

    matrix = np.array(data)
    return matrix

def find_smallest_eigenvector(matrix):
    AtA = np.dot(matrix.T, matrix)

    eigenvalues, eigenvectors = np.linalg.eig(AtA)

    min_eigenvalue_index = np.argmin(eigenvalues)

    smallest_eigenvector = eigenvectors[:, min_eigenvalue_index]

    return smallest_eigenvector

file_path = 'matches2.txt'
matrix = build_matrix(file_path)

np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=6)
print(matrix)
smallest_eigenvector = find_smallest_eigenvector(matrix)

print("Smallest Eigenvector:")
print(smallest_eigenvector)
print(smallest_eigenvector.reshape(3, 3))
print(np.dot(smallest_eigenvector.reshape(3, 3), np.array([915, 251, 1])))