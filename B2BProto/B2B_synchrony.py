from EGG_device1 import EEG
from EGG_device2 import EEG2
from bispectrum import bispec
from timer import timer

# Imports for P300
import multiprocessing
from multiprocessing import Process, Value, Manager

import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
import scipy.stats as stats

# Imports for OpenBCI
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# # CODE FOR REAL TIME TEST # #

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes
seconds = Value("i", 0)
counts = Value("i", 0)


# # Define Parallel Processes # #


###################################################################################################################################################
if __name__ == '__main__':
    # Access to Manager to share memory between proccesses and acces dataframe's 
    mgr = Manager()
    #ns = mgr.list()
    eno1_datach1 = multiprocessing.Array('d', 800)
    eno1_datach2 = multiprocessing.Array('d', 800)


    eno2_datach1 = multiprocessing.Array('d', 800)
    eno2_datach2 = multiprocessing.Array('d', 800)



    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder = 'S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)


    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    #Creación de carpetas para datos de Enophones 2
    for subfolder2 in ['Raw 2', 'Processed 2', 'Figures 2']:
        os.mkdir('{}/{}'.format(folder, subfolder2))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()

    # # Start processes # #

    process2 = Process(target=timer, args=[seconds, counts, timestamps])
    q = Process(target=EEG, args=[seconds, folder, eno1_datach1, eno1_datach2])
    q2 = Process(target=EEG2, args=[seconds, folder, eno2_datach1, eno2_datach2])
    q3 = Process(target=bispec, args=[eno1_datach1, eno1_datach2, eno2_datach1, eno2_datach2, seconds, folder])


    process2.start()
    q.start()
    q2.start()
    q3.start()


    process2.join()
    q.join()
    q2.join()
    q3.join()


    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.

    print(Fore.RED + 'Test finished sucessfully, storing data now...' + Style.RESET_ALL)


    print(Fore.GREEN + 'Data stored sucessfully' + Style.RESET_ALL)

    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)

    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the dataset x, and filters the valid rows back to y.

        :param pd.DataFrame df: with non-normalized, source variables.
        :param string method: type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df.quantile(q=.25)
            q3 = df.quantile(q=.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        
        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')
        return df
    
    # The following for loop iterates over all features, and removes outliers depending on the statistical method used.
    # It reads the files saved in the "Raw" folder, and only reads .CSV files, to outputt a .CSV file in "Processed" folder.
    for df_name in os.listdir('{}/Raw/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    for df_name2 in os.listdir('{}/Raw 2/'.format(folder)):
        if df_name2[-4:] == '.csv' and df_name2[:4] != 'file':
            df_name2 = df_name2[:-4]
            df_raw2 = pd.read_csv('{}/Raw 2/{}.csv'.format(folder, df_name2), index_col=0)
            df_processed2 = remove_outliers(df_raw2.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed2.to_csv('{}/Processed 2/{}_processed2.csv'.format(folder, df_name2))
            df_processed2.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures 2/{}_plot2.png'.format(folder, df_name2))

    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)

                        #Create dataframes to estimate the eyes open mean matrix

    data_meanb = pd.read_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), index_col=0)
    data_graph = data_meanb.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)

                            #matrix = pd.DataFrame(arrange3).transpose()
    #arrange3.to_csv('{}/Calibration_data_clean.csv'.format(folder))

                          
    #eyes_open = pd.read_csv('{}/Calibration_data_clean.csv'.format(folder), index_col=0)
                        
    df_graph = pd.DataFrame(data_graph)
    # Replace -inf and inf with 0 in your DataFrame
    data_graph = data_graph.replace([float('-inf'), float('inf')], 0)
    print(df_graph)

"""
    # Assuming data_graph is your DataFrame
    # Generate a time index from 0 to 420 seconds with the same length as your DataFrame
    time_index = np.linspace(0, 180, len(data_graph))
    print(seconds)
    print(timestamps)
    for column in data_graph.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(time_index, data_graph[column], label=column)
        plt.title(f'Plot of {column}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{folder}/Figures/{column}_plot.png')
        plt.show()


####### Sources ########
# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones



#python Empatica-Project-ALAS-main/files/final_bispec.py
"""