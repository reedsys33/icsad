import sys
import numpy as np
from sklearn import linear_model as lm
from sklearn import svm

np.set_printoptions(precision=5, suppress=True)

''' ----------------------------------------------------------------------------------------------
    --------------------------------------- MANUAL INPUTS ----------------------------------------
    ----------------------------------------------------------------------------------------------       
    Uncomment operation mode (selection is ignored if entered from command line argument):
        -r   selects RANDOM dataset
        -p   selects PROCESS dataset
        -pf  selects PROCESS dataset with anomaly FORCED
        -af  selects ATTACK dataset with additional anomaly FORCED
        -a   selects ATTACK dataset (already contains anomaly) '''    
#MANUAL_MODE = "-r"
#MANUAL_MODE = "-p"
#MANUAL_MODE = "-pf"
#MANUAL_MODE = "-af"
MANUAL_MODE = "-a"

''' Set forcing arameters (only used for -pf and -af modes)
        FORCED_ROW - use 23+, note that -af mode already has an anomaly at 24 embedded
        FORCED_COLUMN - choose 1, 2, 3, 4 or 5 (valid columns in project)
        FORCED_VALUE  - value to be sset as an anomaly (in range of process variable to be realistic)'''
FORCED_ROW    = 23
FORCED_COLUMN = 3
FORCED_VALUE  = 0

''' Set random seed for reproducibility (only used by -r, -p, -pf) '''
RANDOM_SEED   = 36

''' NOTE: to replicate demo from presentation, see commented lines under "Configure dataset for analysis" below '''

''' ----------------------------------------------------------------------------------------------
    ------------------------------------------ FUNCTIONS -----------------------------------------
    ---------------------------------------------------------------------------------------------- '''

def find_median(a, b, c):
    ''' Disregard the high & low, and return the median on 3 passed values. '''
    return sorted([a, b, c])[1]

def generate_random_correlated_data(rows=30, cols=5, base_range=(-100, 100), noise_std=3):
    ''' Produces a matrix (numpy array) with highly correlated columns.
        Optional parameters:
            rows (int): Number of rows in the matrix.
            cols (int): Number of columns in the matrix.
            base_range (tuple): Range of the base random data (min, max).
            noise_std (float): Standard deviation of noise added to each column. '''          
    base_vector = np.random.uniform(base_range[0], base_range[1], rows) #-- create base vector
    matrix = np.array([base_vector + np.random.normal(0, noise_std, rows) for _ in range(cols)]).T
    return matrix

def generate_tank_process_data(lim_dflow=30, lim_tank=1, lim_valve=2, correl=0.25):
    ''' Produces a matrix (numpy array) randomly gerated to represent a Tank Level Control Process.
        Process variables include: In_FLow(GPM), Out_Flow(GPM), Tank_Level(%), Valve_In(%), Valve_Out(%)
        Note: dataset_pf uses this as well, with values later forced to simulate malicious behavior 
        Optional parameters:
            lim_dflow: Change in flow limitation.
            lim_tank: Change in tank level limitation.
            lim_valve: Change in value output limitation.
            correl=0.25: Correlation factor. '''    
    matrix = np.zeros((30, 6)) #-- initialize matrix 
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

''' ----------------------------------------------------------------------------------------------
    ---------------------------------------- MAIN PROGRAM ----------------------------------------
    ---------------------------------------------------------------------------------------------- '''
    
''' Set random seed for reproducibility (only used by -r, -p, -pf) '''
np.random.seed(RANDOM_SEED)

''' Set constants '''
k = 0
verbose = 0
error_sum_max = 0
error_sum_min = 999999999
error_sum_dev = 0.5 #-- default non-zero value for anomaly measurement
anomaly_lim = 1.4 #-- tolerance of error_sum value for anomaly reporting
anomaly_tol = 8 #-- tolerance of error_sum change for anomaly reporting
anomaly_det = 0 #-- reset anomaly detected flag

''' Check if any arguments were passed '''
if len(sys.argv) > 1:
    arg = sys.argv[1:]
    ''' Validate the input argument '''
    if "-v" in arg:
        verbose = 1
    if "-h" in arg:
        print("\n** This script performs State Estimation using 3 forms of recursive regression (Least Squares, Ridge, Support Vector) " +
              "\n   and contains 3 types of datasets with option to force an anomoly. Note that script can be manually configured" + 
              "\n   to select the row, column, and forced value (-pf and -af), as well as random seed to produce data (-r, -p, and -pf) **" +
              "\n\n Command line input choices are:" +
              "\n   -r   selects RANDOM dataset" +
              "\n   -p   selects PROCESS dataset" +
              "\n   -pf  selects PROCESS dataset with anomaly FORCED" +
              "\n   -a   selects ATTACK dataset" +
              "\n   -af  selects ATTACK dataset with additional anomaly FORCED" +
              "\n   -h   displays this HELP menu" +
              "\n   -v   displays VERBOSE report, with itemized detail on error and source")
        sys.exit()
    elif "-r" in arg:
        print("\n** Using RANDOM dataset **")
    elif "-p" in arg:
        print("\n** Using PROCESS dataset **")
    elif "-pf" in arg:
        print("\n** Using PROCESS dataset with anomaly FORCED **")
    elif "-a" in arg:
        print("\n** Using ATTACK dataset **")
    elif "-af" in arg:
        print("\n** Using ATTACK dataset with additional anomaly FORCED **")
    else:
        print("\n** Invalid option or syntax; run with -h for help **")
        sys.exit()
else:
    ''' no arguments were passed, uncomment MANUAL_MODE choice near top of file if not running at command line '''
    arg = [MANUAL_MODE]
    print("\n** Using manually configured dataset by default. Run with -h for more info **")

''' dataset_r variables are 5 columns of random but nighly correlated data between -100 and 100 
    this dataset is a fixed/declared array used during presentation video, 
    need to uncomment dataset_r usage below to enable this '''
dataset_r = np.array([[-95.12,-94.08,-93.57,-96.2,-94.77],
[-88.45,-87.42,-86.89,-89.13,-87.98],
[-70.34,-69.4,-68.88,-71.02,-70.5],
[-50.56,-49.82,-49.01,-51,-50.11],
[-30.67,-29.85,-29.13,-31,-30.05],
[-10.45,-9.32,-9.11,-11,-10.23], #-- Row 6
[10.1,9.2,9.75,8,10.15],
[30.25,29.78,30.1,31,29.5],
[50.15,49.4,50.2,51.3,49.95],
[70.9,69.8,70.5,71,69.55],
[90.1,91.15,90.75,89.9,90.45], #-- Row 11
[-92.8,-91.75,-92.5,-93.2,-91.4],
[-75.6,-76.2,-75.1,-74.85,-75.95],
[-55.1,-54,-54.2,-55.75,-53.9],
[-35.25,-36.15,-35.5,-34.2,-36],
[-15.3,-14.25,-14.85,-15.75,-13.9], #-- Row 16
[5.6,4.8,5.1,6,4.9],
[25.75,24.6,25,25.4,24.9],
[45.9,46.5,45.75,46.1,47],
[65.15,64.8,65.5,66,64.25],
[85.3,86.25,85.75,86.5,84.9], #-- Row 21
[-98.12,-97.05,-97.4,-98,-96.6],
[-80.45,-81.2,-79.9,-80.55,-81.1],
[-60.3,-59.85,-60.1,-61,-59.75],
[-40.2,-39.75,-39.5,-40.8,-41],
[-20.1,-21,-20.5,-19.85,-21.1], #-- Row 26
[0,1.2,0.8,0.25,1.5],
[20.45,19.5,20.1,21,19.75],
[40.1,39.2,40.15,40.5,39.8],
[60.3,61,60.45,60.2,61.5]])

''' dataset_p variables are randomly gerated via Python/Numpy to represent a Tank Level Control Process.
    Process variables include: In_FLow(GPM), Out_Flow(GPM), Tank_Level(%), Valve_In(%), Valve_Out(%)
    Note: dataset_pf is the same as dataset_p, but with Valve_Out(%) forced closed maliciously '''
dataset_p = np.array([[500,500,50,50,50],
[497.43986,500.36867,50.66482,47.73573,51.61658],
[490.37653,506.2167,53.38152,46.27368,53.40368],
[498.60354,520.60951,54.24091,50.04762,58.52882],
[497.28805,521.98701,54.1349,48.99302,59.64333],
[499.34107,519.45932,53.30238,50.17592,59.91242], #-- Row 6
[484.90476,513.77666,54.76716,45.09403,58.49705],
[488.43382,513.56096,53.22414,46.84161,60.26746],
[491.7967,540.43021,58.83818,46.83995,67.56073],
[495.30472,571.758,66.43511,46.44972,75.0881],
[508.61716,578.07693,63.88093,50.12388,77.09341], #-- Row 11
[514.07177,601.03504,68.9327,49.56796,80.90971],
[509.857,598.55397,68.55832,49.83003,79.49574],
[507.96796,600.7476,70.53191,47.37655,80.68484],
[513.8969,632.21234,76.85316,49.57006,87.71134],
[500.70366,593.28127,71.37222,45.35178,78.45063], #-- Row 16
[505.70666,573.51203,65.38885,47.5433,73.22339],
[510.8258,549.84427,58.67066,50.67184,65.84835],
[502.13728,561.11293,62.73833,47.49472,67.85865],
[491.00507,566.67013,66.47629,45.0163,69.52781],
[485.46792,583.33371,71.26687,44.00018,74.05719], #-- Row 21
[481.37924,607.96213,78.03843,43.26702,80.5116],
[483.48514,628.01755,81.76326,42.68582,86.13826],
[481.64319,623.86435,80.82142,44.03633,85.70837],
[496.2944,655.34732,84.85788,47.48763,93.30479],
[484.35575,620.50513,78.26029,45.88861,86.18042], #-- Row 26
[475.62205,620.16669,80.74405,44.5031,85.56606],
[465.46134,588.60718,75.52755,41.15267,77.41964],
[470.05459,619.88056,81.72837,43.55617,86.80568],
[462.65333,613.79021,82.10259,41.29188,86.50787]])

''' dataset_a is time series data taken from a larger dataset containing cybersecurity attacks
    performed on the Water Distribution (WADI) test bed. In attack scenario #6 (from larger dataset,
    upstream MCVs were maliciously turned off representing a disruption of service to consumers.
    Variables below are: Diff Pressure, FCV1(%), FLow1 PV, FCV2, Flow2 PV, where the attack can
    be visually observice by reduction of Flow1 (row 24) and Flow2 (row 25). Data is credited to:
    “iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design”
    Note: dataset_af is the same as dataset_a, but with FCV1(%) & FCV2(%) forced closed maliciously '''
dataset_a = np.array([[1866.69,19.9703,0.469893,13.2592,0.377819],
[1876.13,19.8778,0.47001,13.3274,0.326914],
[1871.82,19.5518,0.482008,13.8492,0.324388],
[1873.47,19.6266,0.469033,14.277,0.325267],
[1887.67,19.3277,0.478283,14.6919,0.326573],
[1896.31,19.0297,0.48464,15.1039,0.324927], #-- Row 6
[1917.48,19.1678,0.454463,15.4375,0.327738],
[1923.18,18.8918,0.46565,15.8658,0.32319],
[1922.08,18.6099,0.464959,16.309,0.319881],
[1927.19,18.2107,0.481382,16.6461,0.328655],
[1499.14,20.1981,0.393389,18.5496,0.268404], #-- Row 11
[1449.75,22.1196,0.388376,20.4368,0.277118],
[1607.43,22.9842,0.428141,21.64,0.301065],
[1703.12,23.0284,0.452463,22.1841,0.318996],
[1736.82,22.6891,0.478567,22.7402,0.318464],
[1750.55,23.0318,0.442495,22.9855,0.328329], #-- Row 16
[1754.16,22.8162,0.454193,23.3732,0.324854],
[1759.26,22.231,0.49163,23.0079,0.362137],
[1763.24,22.5365,0.446606,22.149,0.373631],
[1758.53,22.1028,0.483856,21.6135,0.351308],
[1762.26,22.4353,0.442544,20.6531,0.369279], #-- Row 21
[1758.91,21.9993,0.475425,20.048,0.361442],
[1762.91,21.7982,0.479978,19.4825,0.351846],
[1912.34,30.4548,0.0403264,18.0273,0.390842],
[2130.37,42.2175,0.0403264,25.3097,0.0615564],
[2217.02,53.9817,0.0403264,33.1149,0.0615564], #-- Row 26
[2222.63,65.9585,0.0403264,41.0599,0.0615564],
[2230.55,77.7219,0.0403264,48.8656,0.0615564],
[2229.95,89.2762,0.0403264,56.5311,0.0615564],
[2232.75,100,0.0403264,64.1948,0.0615564]])


''' Configure dataset for analysis '''
if "-r" in arg:
    matrix = generate_random_correlated_data() #-- selects RANDOM dataset (via function)
    #matrix = np.array(dataset_r) #-- selects RANDOM dataset (from presentation)
elif "-p" in arg:
    matrix = generate_tank_process_data() #-- selects PROCESS dataset (via function)
    #matrix = np.array(dataset_p) #-- selects PROCESS dataset (from presentation)
elif "-pf" in arg:
    matrix = generate_tank_process_data() #-- selects PROCESS dataset (via function)
    #matrix = np.array(dataset_p) #-- selects PROCESS dataset (from presentation)
    matrix[FORCED_ROW-1:, FORCED_COLUMN-1] = FORCED_VALUE #-- uses MANUAL INPUTS for forced anomaly
    #matrix[24:, 4] = 0 #-- sets columnn 5 to zero for forced anomaly (from presentation)
elif "-af" in arg:
    matrix = np.array(dataset_a) #-- selects ATTACK dataset before anomaly is FORCED
    matrix[FORCED_ROW-1:, FORCED_COLUMN-1] = FORCED_VALUE #-- uses MANUAL INPUTS for forced anomaly
    #matrix[22:, 1] = 0 #-- sets columnn 2 to zero for forced anomaly (from presentation)
    #matrix[23:, 3] = 0 #--  sets columnn 4 to zero for forced anomaly (from presentation)
elif "-a" in arg:
    matrix = np.array(dataset_a) #-- selects ATTACK dataset (anomaly is in original data)
else:
    print("\n** Exception: dataset not assigned; run with -h for help **")
    sys.exit()

if verbose == 1:
    print("--------|----------------------------------------------------------")
    print("   X,Y  |   STATE ESTIMATION ERROR DATA")
    print("--------|----------------------------------------------------------")

    
''' Initialize '''
regr_len = 9 #-- 8 rows for analysis & 1 to estimate and compare
data_len = len(matrix) #-- available data for algorithm
regr_type = ''
error_matrix = []
error_sum_dlast = 0
error_sum_dlast2 = 0

''' Ensure enough data is available, then iterate through each row '''
if data_len > regr_len:
    for c_row in range(regr_len-1, data_len):
        error_row = []
        ''' Iterate through each column '''
        for c_col in range(0, 5):
            obs_value = matrix[c_row, c_col]
         
            ''' Prepare data for regression analysis, separate into features (X) and target (y) ''' 
            X = np.delete(matrix[(c_row+1-regr_len):(c_row+1), :], c_col, axis=1) #-- regr_len of rows, except the estimated column
            y = matrix[(c_row+1-regr_len):(c_row+1), c_col]  #-- regr_len of rows, estimated column only
            y_range = np.ptp(y) #-- determines range of estimated column for deviation calculations
            
            ''' Remove the last row of data for regression analysis ''' 
            X_train = np.delete(X, regr_len-1, axis=0)
            y_train = np.delete(y, regr_len-1)
        
            ''' Perform 3 different types of regression analysis ''' 
            model1 = lm.LinearRegression()
            model2 = lm.Ridge()
            model3 = svm.SVR(kernel="linear", C=0.01, gamma="auto")
            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)
            
            ''' Estimate the value of the target variable and check deviations'''
            est_value1 = model1.predict(np.array([np.delete(matrix[c_row, :], c_col, axis=None)]))
            est_value2 = model2.predict(np.array([np.delete(matrix[c_row, :], c_col, axis=None)]))
            est_value3 = model3.predict(np.array([np.delete(matrix[c_row, :], c_col, axis=None)]))     
            dev1 = round((abs(obs_value - est_value1[0]) / y_range), 3)
            dev2 = round((abs(obs_value - est_value2[0]) / y_range), 3)
            dev3 = round((abs(obs_value - est_value3[0]) / y_range), 3)
            
            ''' Choose median result to eliminate outlier estimations '''
            med_value = find_median(est_value1[0], est_value2[0], est_value3[0])
            ''' Store analysis type for this data point '''
            if med_value == est_value1:
                regr_type = "Least Squares Regression"
                dev = dev1
            elif med_value == est_value2:
                regr_type = "Ridge Regression"
                dev = dev2
            elif med_value == est_value3:
                regr_type = "Support Vector Regression"
                dev = dev3
        
            ''' If VERBOSE option selected, print itemized detail of error and source '''
            if verbose == 1:
                print(f"  {c_row:02},{c_col:02} | LSR {dev1:.3f}, RR {dev2:.3f}, SVR {dev3:.3f}, ==> {regr_type}")
                #print(f"   Error: {dev}% | Observed: {obs_value} | Estimated: {est_value} | LSR {dev1}%, RR {dev2}%, SVR {dev3}%, ==> {regr_type}") #-- alternate verbose
            
            error_row.append(dev.item())
        error_matrix.append(error_row)
    
    if verbose == 1:
        print()
    
    ''' display chosen dataset and include any forced values '''
    print("--------|----------------------------------------------------------")
    print("  ROW   |  ORIGINAL DATASET")
    print("--------|----------------------------------------------------------")
    for j in range(0, len(matrix)):
        print(f"  [{(j+1):02}]  |  {matrix[j]}")
    
    ''' display information on anomaly metric and state estimation errors '''
    error_row = regr_len
    print("\n--------|-----------|----------------------------------------------")
    print("  ROW   |  ANOMALY  |  ESTIMATION ERRORS")
    print("--------|-----------|----------------------------------------------")
    for row in error_matrix:
        ''' sum the errors in each row, and track the range of sums '''
        error_sum = round(np.sum(row), 3)
        error_sum_range = error_sum_max - error_sum_min
        error_sum_dlast2 = error_sum_dlast
        error_sum_dlast = error_sum_dev
                
        ''' need data to determine a range, then calculate deviation as sum / sum_range '''
        if k > 1: 
            error_sum_dev = round(error_sum / error_sum_range, 3) #-- anomaly measurement
        else:
            error_sum_dev = 0.5 #-- default non-zero value for anomaly measurement
        if error_sum < error_sum_min:
            error_sum_min = error_sum #-- update range
        if error_sum > error_sum_max:
            error_sum_max = error_sum #-- update range
    
        ''' ignore initial rows of comparison, but print line with no anomaly measurement '''
        if k >= 2 and k <= 11:
            print(f"  [{error_row}]  |  [-n/a-]  |  {row}")
        ''' if remaining rows show anomaly measurment changing faster than acceptable tolerance, flag row '''
        if k > 11:
            if (error_sum_dev>anomaly_lim) & ((error_sum_dev/error_sum_dlast)>anomaly_tol or 
                                              (error_sum_dev/error_sum_dlast2)>anomaly_tol):
                print(f"**[{error_row}]**|**[{error_sum_dev:.3f}]**|**{row}**********ANOMALY IDENTIFIED**********")
                anomaly_det = 1
            else:
                print(f"  [{error_row}]  |  [{error_sum_dev:.3f}]  |  {row}")
        k += 1        
        error_row += 1
    
    if anomaly_det == 1:
        print("\n^^^ Anomaly identified in dataset ^^^")
    else:
        print("\n--- No issues found ---")