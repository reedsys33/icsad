import numpy as np

''' Set random seed for reproducibility '''
np.random.seed(0)

def generate_tank_process_data(lim_dflow=30, lim_tank=1, lim_valve=2, correl=0.25):
    ''' Produces a matrix (numpy array) randomly gerated to represent a Tank Level Control Process.
        Process variables include: In_FLow(GPM), Out_Flow(GPM), Tank_Level(%), Valve_In(%), Valve_Out(%)
        Note: dataset_pf uses this as well, with values later forced to simulate malicious behavior 
        Optional parameters:
            lim_dflow: Change in flow limitation.
            lim_tank: Change in tank level limitation.
            lim_valve: Change in value output limitation.
            correl=0.25: Correlation factor. '''    
    
    ''' Initialize matrix '''
    matrix = np.zeros((30, 6))
    
    ''' Column 1: Random values starting from 0, range [-250, 250], max step change of 50 '''
    matrix[0, 0] = 0
    for i in range(1, 30):
        step = np.random.uniform(-lim_dflow, lim_dflow)
        matrix[i, 0] = np.clip(matrix[i-1, 0] + step, -250, 250)
    
    ''' Column 2: Random values starting from 500, range [0, 1000], max step change of 25 '''
    matrix[0, 1] = 500
    for i in range(1, 30):
        step = np.random.uniform(-lim_dflow/2, lim_dflow/2)
        matrix[i, 1] = np.clip(matrix[i-1, 1] + step, 0, 1000)
    
    ''' Column 3: Sum of columns 1 and 2, range [0, 1000] '''
    matrix[:, 2] = np.clip(matrix[:, 0] + matrix[:, 1], 0, 1000)
    
    ''' Column 4: Random values starting from 80, highly correlated to column 1, max step change of 1.0 '''
    matrix[0, 3] = 50
    for i in range(1, 30):
        correlation_factor = correl * (matrix[i, 0] - matrix[i-1, 0])  # small correlation to column 1
        step = np.random.uniform(-lim_tank, lim_tank) + correlation_factor
        matrix[i, 3] = np.clip(matrix[i-1, 3] + step, 0, 100)
    
    ''' Column 5: Random values starting from 50, highly correlated to column 2, max step change of 2.5 '''
    matrix[0, 4] = 50
    for i in range(1, 30):
        correlation_factor = correl * (matrix[i, 1] - matrix[i-1, 1])  # small correlation to column 2
        step = np.random.uniform(-lim_valve, lim_valve) + correlation_factor
        matrix[i, 4] = np.clip(matrix[i-1, 4] + step, 0, 100)
    
    ''' Column 6: Random values starting from 50, highly correlated to column 3, max step change of 2.5 '''
    matrix[0, 5] = 50
    for i in range(1, 30):
        correlation_factor = correl * (matrix[i, 2] - matrix[i-1, 2])  # small correlation to column 3
        step = np.random.uniform(-lim_valve, lim_valve) + correlation_factor
        matrix[i, 5] = np.clip(matrix[i-1, 5] + step, 0, 100)
    
    ''' Return the resulting matrix (remove first column which was used to create both flows) '''
    return(np.delete(matrix, [0], axis=1))

''' Call function '''
matrix = generate_tank_process_data()

''' Display the result '''
print("Generated Matrix:")
print(matrix)

''' Uncomment to save data to a csv file '''
#np.savetxt("tank_process_data.csv", matrix, delimiter=",", header="Col1,Col2,Col3,Col4,Col5", comments="")






