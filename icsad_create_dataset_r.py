import numpy as np

''' Set random seed for reproducibility '''
np.random.seed(0)

def generate_random_correlated_data(rows=30, cols=5, base_range=(-100, 100), noise_std=3):
    ''' Produces a matrix (numpy array) with highly correlated columns.
        Optional parameters:
            rows (int): Number of rows in the matrix.
            cols (int): Number of columns in the matrix.
            base_range (tuple): Range of the base random data (min, max).
            noise_std (float): Standard deviation of noise added to each column. '''
            
    ''' Generate a base vector '''
    base_vector = np.random.uniform(base_range[0], base_range[1], rows)
    
    ''' Create columns by adding small noise to the base vector '''
    matrix = np.array([base_vector + np.random.normal(0, noise_std, rows) for _ in range(cols)]).T
    return matrix

''' Call function '''
matrix = generate_random_correlated_data()

''' Display the result '''
print("Generated Matrix:")
print(matrix)

''' Uncomment to save data to a csv file '''
#np.savetxt("random_correlated_data.csv", matrix, delimiter=",", header="Col1,Col2,Col3,Col4,Col5", comments="")
