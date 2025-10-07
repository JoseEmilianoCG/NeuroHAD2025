from EGG_device1 import EEG
from EGG_device2 import EEG2
from bispectrum import bispec
from timer import timer
from markers import marker_loop
from neuro_dashboard import run_dash, push, shutdown

# Imports for P300
import multiprocessing
from multiprocessing import Queue, Process, Value, Manager

import numpy as np
from colorama import Fore, Style
import scipy.stats as stats

# Imports for OpenBC
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BoardIds

# # CODE FOR REAL TIME TEST # #99

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes
seconds = Value("i", 0)
counts = Value("i", 0)

# Time parameters
basaltime = 30
totaltime = 600
sleeptime = 2

# Sampling rate
sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
windsz = int(sleeptime * sampling_rate)


# Marker labels
labels = {'1':'baseline_start', '2':'shared_start', '3':'individual_start', ' ':'event'}

# # Define Parallel Processes # #


###################################################################################################################################################
if __name__ == '__main__':
    # Access to Manager to share memory between proccesses and acces dataframe's 
    mgr = Manager()
    #ns = mgr.list()
    eno1_datach1 = multiprocessing.Array('d', windsz)
    eno1_datach2 = multiprocessing.Array('d', windsz)
    eno1_datach3 = multiprocessing.Array('d', windsz)
    eno1_datach4 = multiprocessing.Array('d', windsz)


    eno2_datach1 = multiprocessing.Array('d', windsz)
    eno2_datach2 = multiprocessing.Array('d', windsz)
    eno2_datach3 = multiprocessing.Array('d', windsz)
    eno2_datach4 = multiprocessing.Array('d', windsz)

    dash_q = Queue()



    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    # --- Solicitud de sujeto y repetición ---
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = f'{int(subject_ID):02d}'
    repetition_num = f'{int(repetition_num):02d}'

    # --- Ruta base absoluta (carpeta donde está este script) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'Outputs')

    # Crea Outputs si no existe
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --- Carpeta del sujeto ---
    folder_name = f"S{subject_ID}R{repetition_num}_{datetime.now().strftime('%d%m%Y_%H%M')}"
    folder = os.path.join(OUTPUTS_DIR, folder_name)

    # Crea carpeta principal del sujeto
    os.makedirs(folder, exist_ok=True)

    # --- Subcarpetas principales ---
    for subfolder in ['Prepro', 'Processed', 'Figures']:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # --- Subcarpetas de Enophones 2 ---
    for subfolder2 in ['Prepro 2', 'Processed 2', 'Figures 2']:
        os.makedirs(os.path.join(folder, subfolder2), exist_ok=True)

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = mgr.list()

    # # Start processes # #

    p_dash = Process(
        target=run_dash,
        args=(dash_q,),
        kwargs=dict(port=8051, window_sec=60, xmin=30, title='Impacto de la Lectura Compartida en Sincronización Biométrica y Conexión Emocional'),
        daemon=True
    )
    p_dash.start()
    process2 = Process(target=timer, args=[seconds, counts, timestamps, totaltime])
    q = Process(target=EEG, args=[seconds, folder, eno1_datach1, eno1_datach2, eno1_datach3, eno1_datach4, totaltime])
    q2 = Process(target=EEG2, args=[seconds, folder, eno2_datach1, eno2_datach2, eno2_datach3, eno2_datach4, totaltime])
    q3 = Process(target=bispec, args=[eno1_datach1, eno1_datach2, eno1_datach3, eno1_datach4, eno2_datach1, eno2_datach2, eno2_datach3, eno2_datach4, seconds, basaltime, totaltime, sleeptime, folder, dash_q])
    q4 = Process(target=marker_loop, args=(f'{folder}/markers.csv', labels), daemon=True)



    # process2.start()
    # q.start()
    # q2.start()
    # q3.start()
    # q4.start()


    # process2.join()
    # q.join()
    # q2.join()
    # q3.join()

    try:
        process2.start()
        q.start()
        q2.start()
        q3.start()
        q4.start()

        process2.join()
        q.join()
        q2.join()
        q3.join()
    finally:
        # Cierre limpio del dashboard (y kill de respaldo si siguiera vivo)
        shutdown(dash_q)
        try:
            if p_dash.is_alive():
                p_dash.terminate()
        except:
            pass


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
    for df_name in os.listdir('{}/Prepro/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Prepro/{}.csv'.format(folder, df_name), index_col=0).drop(['board_ts','unix_ts'],axis=1)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')

            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    for df_name2 in os.listdir('{}/Prepro 2/'.format(folder)):
        if df_name2[-4:] == '.csv' and df_name2[:4] != 'file':
            df_name2 = df_name2[:-4]
            # Uncomment for muse 2
            # df_raw2 = pd.read_csv('{}/Prepro 2/{}.csv'.format(folder, df_name2), index_col=0).drop(['board_ts','unix_ts'],axis=1)
            df_raw2 = pd.read_csv('{}/Prepro 2/{}.csv'.format(folder, df_name2), index_col=0)[['Fz', 'C3', 'Cz', 'C4']] # Synthetic only
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