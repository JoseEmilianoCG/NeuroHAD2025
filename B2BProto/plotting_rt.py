from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, FilterTypes, DetrendOperations
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations
import time

# This object is created to display in real time ENOPHONE's data, using self as argument, a variable that can be updated is created
# The argument board allow to call the specific board we are using in the main function to establish the connection.
class Graph:
    def __init__(self, board, eeg_channels, sampling_rate, window_duration):
        #Data parameters to establish connections
        self.board_id = board.get_board_id()
        self.board_shim = board
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.update_speed_ms = int(window_duration * 1000)
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        #Calling the app inicialization to create a new window with plots
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="BOARD 1")

        # Functions inside the object to arrange data
        self._init_timeseries()

        # Tools to start running the plot in real time
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()


    #Plot for raw data
    def _init_timeseries(self):
        # Create an empty list to update data once the code start running
        self.plots = list()
        self.curves = list()
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Raw Data')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()

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
board.start_stream(45000)
BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Enophones ---')


previous_time = time.time()
while (True):
    time.sleep(4)

    current_time = time.time()
    window_duration = current_time - previous_time
    previous_time = current_time

    print(f"Window duration: {window_duration:.2f} seconds")

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
    print(df_crudas)

    #Uncomment the line below if you want to se the real time graphics of the preprocessing stage, it may cause problems in code efficiency and shared memory
    Graph(board, eeg_channels, sampling_rate, window_duration)