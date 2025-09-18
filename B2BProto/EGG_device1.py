import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations
import time
from plotting import Graph

# # CODE FOR EEG # #
def EEG(second, folder, eno1_datach1, eno1_datach2):
    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    #params.mac_address = 'f4:0e:11:75:75:a5'

    # Relevant board IDs available:
    #board_id = BoardIds.ENOPHONE_BOARD.value # (37)
    board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)

    # An empty dataframe is created to save Alpha/Beta values to plot in real time.
    #alpha_beta_data = pd.DataFrame(columns=['Alpha_C' + str(c) for c in range(1, len(eeg_channels) + 1)])
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Enophones ---')

    try:
        while (True):
            time.sleep(4)
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            signal = pd.DataFrame(columns=['CH' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            
            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                signal['CH' + str(eeg_channel)] = data[eeg_channel]
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                ####################START OF PREPROCESING#############################
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_crudas['MV' + str(eeg_channel)] = data[eeg_channel]
            
            signal.to_csv('{}/Raw/Testing.csv'.format(folder), mode='a')
            df_crudas.to_csv('{}/Raw/Crudas.csv'.format(folder), mode='a')
            
            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_crudas['MV3'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_crudas['MV4'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            arrange=referenced_electrodes.to_dict('dict')

            
            info1=arrange['referenced_electrode1']
            info2=arrange['referenced_electrode2']
            #info3=arrange['MV3']
            #info4=arrange['MV4']

            lista1 = list(info1.values())
            lista2 = list(info2.values())
            #lista3 = list(info3.values())
            #lista4 = list(info4.values())

            eno1_datach1[:800] = lista1[:800]
            eno1_datach2[:800] = lista2[:800]
            #eno1_datach3[:800] = lista3[:800]
            #eno1_datach4[:800] = lista4[:800]


            #Uncomment the line below if you want to se the real time graphics of the preprocessing stage, it may cause problems in code efficiency and shared memory
            #Graph(board)
            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value == 21):
                    return      

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophones ---')

    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.