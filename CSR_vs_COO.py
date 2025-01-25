import numpy as np
import time

# Coordinate
class COOMatrix:
    def __init__(self, row_indices, col_indices, values, shape):
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.values = values
        self.shape = shape
    
    def matrix_vector_mult(self, vector):
        result = np.zeros(self.shape[0])
        # Must check each non-zero element individually
        for i in range(len(self.values)):
            row, col = self.row_indices[i], self.col_indices[i]
            result[row] += self.values[i] * vector[col]
        return result

# Compressed Sparse Row
class CSRMatrix:
    def __init__(self, values, col_indices, row_ptr, shape):
        self.values = values
        self.col_indices = col_indices
        self.row_ptr = row_ptr
        self.shape = shape
    
    def matrix_vector_mult(self, vector):
        result = np.zeros(self.shape[0])
        # Process each row's elements contiguously
        for i in range(self.shape[0]):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            # Process entire row at once with good memory locality
            for j in range(start, end):
                result[i] += self.values[j] * vector[self.col_indices[j]]
        return result

# Compressed Sparse Column
class CSCMatrix:
    def __init__(self, values, row_indices, col_ptr, shape):
        self.values = values
        self.row_indices = row_indices
        self.col_ptr = col_ptr
        self.shape = shape

    def matrix_vector_mult(self, vector):
        result = np.zeros(self.shape[0])
        # Process each col's elements contiguously
        for j in range(self.shape[1]):
            start, end = self.col_ptr[j], self.col_ptr[j+1]
            # Process parts of rows with bad memory locality
            for i in range(start, end):
                result[self.row_indices[i]] = self.values[i] * vector[j]
        return result 


# Create a large sparse matrix (0.1% density)
size = 10000
nnz = size * size // 100  # 0.1% non-zero elements

# Generate random sparse matrix
np.random.seed(42)
row_indices = np.random.randint(0, size, nnz)
col_indices = np.random.randint(0, size, nnz)
values = np.random.randn(nnz)

# Sort indices for CSR conversion
sort_idx = np.lexsort((col_indices, row_indices))
row_indices = row_indices[sort_idx]
col_indices = col_indices[sort_idx]
values = values[sort_idx]

# Create row pointers for CSR
row_counts = np.bincount(row_indices, minlength=size)
row_ptr = np.zeros(size + 1, dtype=np.int32)
np.cumsum(row_counts, out=row_ptr[1:])

# Create matrices
coo_matrix = COOMatrix(row_indices, col_indices, values, (size, size))
csr_matrix = CSRMatrix(values, col_indices, row_ptr, (size, size))

# Test vector
vector = np.random.randn(size)

# Time the operations
def time_operation(matrix, vector, num_trials=30):
    times = []
    for _ in range(num_trials):
        start = time.time()
        result = matrix.matrix_vector_mult(vector)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)

print("Testing with matrix size:", size, "x", size)
print("Number of non-zeros:", nnz)
print("\nTiming matrix-vector multiplication:")

coo_time, coo_std = time_operation(coo_matrix, vector)
print(f"COO format: {coo_time:.4f}s ± {coo_std:.4f}s")

csr_time, csr_std = time_operation(csr_matrix, vector)
print(f"CSR format: {csr_time:.4f}s ± {csr_std:.4f}s")

print(f"\nCSR speedup: {coo_time/csr_time:.2f}x")