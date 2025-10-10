import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations
import time

def pad_or_trim(seq, target_len, pad_value=0.0):
    m = len(seq)
    if m >= target_len:
        # nos quedamos con las últimas 'target_len' muestras
        return list(seq)[-target_len:]
    # left-pad: ceros al inicio para que lo más reciente quede al final
    return [pad_value] * (target_len - m) + list(seq)

# # CODE FOR EEG # #
def EEG2(second, folder, eno2_datach1, eno2_datach2, eno2_datach3, eno2_datach4, totaltime, muse_id):
    print(muse_id)
    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    #params.mac_address = 'f4:0e:11:75:75:ce'
    params.serial_number = muse_id #'Muse-E215'#'Muse-070E'##'

    # Relevant board IDs available:
    #board_id = BoardIds.ENOPHONE_BOARD.value # (37)
    #board_id = BoardIds.SYNTHETIC_BOARD.value
    board_id = BoardIds.MUSE_2_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    channel_names = BoardShim.get_board_descr(board_id)['eeg_names'].split(',')
    ts_ch = BoardShim.get_timestamp_channel(board_id) # Timestamps channel
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)

    # An empty dataframe is created to save Alpha/Beta values to plot in real time.
    #alpha_beta_data = pd.DataFrame(columns=['Alpha_C' + str(c) for c in range(1, len(eeg_channels) + 1)])
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI2.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Enophones 2---')

    try:
        while (True):
            time.sleep(4)
            t_pull = time.time()
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ## SOLO PARA PRUEBAS CON MUSE + SYNTHETIC ##
             
            eeg_channels = eeg_channels[:4] # los primeros 4

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_clean = pd.DataFrame(columns=channel_names + ['board_ts','unix_ts'])
            df_raw = pd.DataFrame(columns=channel_names + ['board_ts','unix_ts'])

            board_ts = data[ts_ch]  # timestamp de BrainFlow (por muestra)
            unix_ts = t_pull - (len(board_ts)-1)/sampling_rate + np.arange(len(board_ts))/sampling_rate

            df_raw['board_ts']  = board_ts
            df_raw['unix_ts']   = unix_ts
            df_clean['board_ts'] = board_ts
            df_clean['unix_ts']  = unix_ts


            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                df_raw[channel_names[eeg_channel - 1]] = data[eeg_channel]
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                ####################START OF PREPROCESING#############################
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_clean[channel_names[eeg_channel - 1]] = data[eeg_channel]
                
            df_raw.to_csv('{}/Prepro 2/Raw2.csv'.format(folder), mode='a')
            df_clean.to_csv('{}/Prepro 2/Clean2.csv'.format(folder), mode='a')
            
            # Calculate the new variable based on the formula
            # referenced_electrodes = pd.DataFrame()
            # referenced_electrodes['referenced_electrode1'] = df_clean['MV3'] - ((df_clean['MV1'] + df_clean['MV2']) / 2)
            # referenced_electrodes['referenced_electrode2'] = df_clean['MV4'] - ((df_clean['MV1'] + df_clean['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            # arrange=referenced_electrodes.to_dict('dict')
            arrange = df_clean.to_dict('dict')

            
            # Case of re-referenced electrodes only
            # info1=arrange['referenced_electrode1']
            # info2=arrange['referenced_electrode2']

            info1 = arrange[channel_names[0]]
            info2 = arrange[channel_names[1]]
            info3 = arrange[channel_names[2]]
            info4 = arrange[channel_names[3]]

            lista1 = list(info1.values())
            lista2 = list(info2.values())
            lista3 = list(info3.values())
            lista4 = list(info4.values())

            window = len(eno2_datach1)
  
            # eno2_datach1[:800] = lista1[:800]
            # eno2_datach2[:800] = lista2[:800]
            # eno2_datach3[:800] = lista32[:800]
            # eno2_datach4[:800] = lista42[:800]

            eno2_datach1[:] = pad_or_trim(lista1, window)
            eno2_datach2[:] = pad_or_trim(lista2, window)
            eno2_datach3[:] = pad_or_trim(lista3, window)
            eno2_datach4[:] = pad_or_trim(lista4, window)


            #Uncomment the line below if you want to se the real time graphics of the preprocessing stage, it may cause problems in code efficiency and shared memory
            # Graph2(board)
            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value == totaltime):
                    return

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophone 2 ---')

    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.