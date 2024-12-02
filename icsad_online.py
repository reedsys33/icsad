import numpy as np
from sklearn import linear_model as lm
from sklearn import svm
from scapy.all import sniff
from scapy.all import struct
from scapy.all import TCP

np.set_printoptions(precision=5, suppress=True)

''' Set constants '''
last_timestamp = 0
begin_detection = False
detection_complete = False
captured_data = [[0, 0, 0, 0, 0, 0]]
numpy_data = [[0, 0, 0, 0, 0]]

''' Function to accept 2 words of data (from Modbus TCP packet) and reproduce the floating point value that they represent '''
def words_to_real(word1, word2):
    # Converts two 16-bit words into a a 32-bit DWORD.
    dword_value = (word1 << 16) | word2
    # Pack the DWORD value as a 4-byte unsigned integer, then unpack it as a signle-precision float
    real_value = struct.unpack('f', struct.pack('I', dword_value))[0]
    return real_value

''' Function to find the median of 3 values (used for automated selection of regression algorithm '''
def find_median(a, b, c):
    return sorted([a, b, c])[1]

''' Function to extract Modbus data from TCP packets '''
def extract_modbus_data(packet):
    global begin_detection, detection_complete, numpy_data
    # Check if the packet has a TCP layer
    if TCP in packet:
        # Ensure the payload is at least 7 bytes (minimum Modbus PDU size)
        if len(packet[TCP].payload) >= 7:
            modbus_data = bytes(packet[TCP].payload)
            if len(modbus_data) >= 18:
                # set mbts to most recent packet timestamp
                mbts = int(modbus_data.hex()[30:34],16)
                # if last saved row differs from latest packet timestamp, save/print new entry
                if mbts != captured_data[-1][0]:
                    b = 34
                    mbreal1  = words_to_real(int(modbus_data.hex()[b:b+4],16),int(modbus_data.hex()[b+4:b+8],16))
                    b = 42
                    mbreal2  = words_to_real(int(modbus_data.hex()[b:b+4],16),int(modbus_data.hex()[b+4:b+8],16))
                    b = 50
                    mbreal3  = words_to_real(int(modbus_data.hex()[b:b+4],16),int(modbus_data.hex()[b+4:b+8],16))
                    b = 58
                    mbreal4  = words_to_real(int(modbus_data.hex()[b:b+4],16),int(modbus_data.hex()[b+4:b+8],16))
                    b = 66
                    mbreal5  = words_to_real(int(modbus_data.hex()[b:b+4],16),int(modbus_data.hex()[b+4:b+8],16))
                    
                    last_timestamp = mbts 
                    if begin_detection == False:
                        #print(f"TxID: {int(modbus_data.hex()[0:4],16)}, Ts: {mbts:02}")
                        print(".", end="", flush=True)
                    
                    if last_timestamp == 1:
                        print()
                        print()
                        #deleted the initialization row
                        numpy_data = np.delete(numpy_data, [0], axis=0)
                        begin_detection = True
                    
                    if begin_detection == True:
                        print(f"TxID: {int(modbus_data.hex()[0:4],16)}, Ts: {mbts:02}, Hex Modbus Data: {modbus_data.hex()[26:]}")                
                        captured_data.append([mbts, mbreal1, mbreal2, mbreal3, mbreal4, mbreal5])
                        numpy_data = np.vstack((numpy_data, [mbreal1, mbreal2, mbreal3, mbreal4, mbreal5]))

                        matrix = np.array(numpy_data) #-- line added to align sniffer program with offline detection algorithm
                        
                        ''' Set constants '''
                        k = 0
                        error_sum_max = 0
                        error_sum_min = 999999999
                        error_sum_dev = 0.5 #-- default non-zero value for anomaly measurement
                        anomaly_lim = 1.4 #-- tolerance of error_sum value for anomaly reporting
                        anomaly_tol = 8 #-- tolerance of error_sum change for anomaly reporting
                        
                        ''' Initialize '''
                        regr_len = 9 #-- 8 rows for analysis & 1 to estimate and compare
                        data_len = len(matrix) #-- available data for algorithm
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
                                    ''' Store analysis type deviation value for this data point '''
                                    if med_value == est_value1:
                                        # Least Squares Regression
                                        dev = dev1
                                    elif med_value == est_value2:
                                        # Ridge Regression
                                        dev = dev2
                                    elif med_value == est_value3:
                                        # Support Vector Regression
                                        dev = dev3
                                        
                                    error_row.append(dev.item())
                                error_matrix.append(error_row)
                            
                            ''' Display information on anomaly metric and state estimation errors '''
                            error_row = regr_len                            
                            if mbts > 10:
                                print("--------|------------------------------------------------------------|-----------|----------------------------------------------")
                                print("  ROW   |  COLLECTED DATASET                                         |  ANOMALY  |  ESTIMATION ERRORS")
                                print("--------|------------------------------------------------------------|-----------|----------------------------------------------")

                            for row in error_matrix:
                                ''' Sum the errors in each row, and track the range of sums '''
                                error_sum = round(np.sum(row), 3)
                                error_sum_range = error_sum_max - error_sum_min
                                error_sum_dlast2 = error_sum_dlast
                                error_sum_dlast = error_sum_dev
                                
                                ''' Need data to determine a range, then calculate deviation as sum / sum_range '''
                                if k > 1: 
                                    error_sum_dev = round(error_sum / error_sum_range, 3) #-- anomaly measurement
                                else:
                                    error_sum_dev = 0.5 #-- default non-zero value for anomaly measurement
                                if error_sum < error_sum_min:
                                    error_sum_min = error_sum #-- update range
                                if error_sum > error_sum_max:
                                    error_sum_max = error_sum #-- update range
                            
                                ''' Ignore initial row of comparison, but print line with no anomaly measurement '''
                                if k >= 2 and k <= 11:
                                    print(f"  [{error_row}]  |  {matrix[error_row-1]}  |  [-n/a-]  |  {row}")
                                    
                                ''' If remaining rows show anomaly measurment changing faster than acceptable tolerance, flag row '''
                                if k > 11:
                                    if (error_sum_dev>anomaly_lim) & ((error_sum_dev/error_sum_dlast)>anomaly_tol or 
                                                                      (error_sum_dev/error_sum_dlast2)>anomaly_tol):
                                        print(f"**[{error_row}]**|**{matrix[error_row-1]}**|**[{error_sum_dev:.3f}]**|**{row}**********ANOMALY IDENTIFIED**********")
                                    else:
                                        print(f"  [{error_row}]  |  {matrix[error_row-1]}  |  [{error_sum_dev:.3f}]  |  {row}")
                                k += 1        
                                error_row += 1
                            print()
                        
                        ''' When last expected timestamp is found, display full dataset collected and exit '''
                        if last_timestamp == 30:
                            print("Targeted data acquired, packet intercept ended.\n")
                            
                            print("Displaying full dataset collected...")
                            print("--------|----------------------------------------------------------")
                            print("  ROW   |  COLLECTED DATASET")
                            print("--------|----------------------------------------------------------")
                            for j in range(0, len(matrix)):
                                print(f"  [{(j+1):02}]  |  {matrix[j]}")
                            print()
                            detection_complete = True
                            return

''' Begin packet inspection, then call "extract_modbus_data" function above, and loop until "detection_complete" is True '''
print("\nSniffing Modbus TCP packets...\n")

print("Looking for Timestamp 01 to begin data collection (anomaly detection results begin at Ts 11).", end="", flush=True)

sniff(iface="enp1s0", filter="src 192.168.70.21 and dst 192.168.70.31", prn=extract_modbus_data, stop_filter=lambda x: detection_complete==True, store=0)
