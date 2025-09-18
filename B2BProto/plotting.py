from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, FilterTypes, DetrendOperations

# This object is created to display in real time ENOPHONE's data, using self as argument, a variable that can be updated is created
# The argument board allow to call the specific board we are using in the main function to establish the connection.
class Graph:
    def __init__(self, board):
        #Data parameters to establish connections
        self.board_id = board.get_board_id()
        self.board_shim = board
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 2000
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        #Calling the app inicialization to create a new window with plots
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="BOARD 1")

        # Functions inside the object to arrange data
        self._init_timeseries()
        self._init_processed()

        # Tools to start running the plot in real time
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.timeout.connect(self.preproccesing)
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
    #Plot for processed data
    def _init_processed(self):
        self.plots2 = list()
        self.curves2 = list()
        for i in range(len(self.eeg_channels)):
            p2 = self.win.addPlot(row=i, col=1)
            p2.showAxis('left', False)
            p2.setMenuEnabled('left', False)
            p2.showAxis('bottom', False)
            p2.setMenuEnabled('bottom', False)
            if i == 0:
                p2.setTitle('Processed Signal')
            self.plots2.append(p2)
            curve2 = p2.plot()
            self.curves2.append(curve2) 

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


    def preproccesing(self):
        data2 = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.detrend(data2[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data2[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data2[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data2[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves2[count].setData(data2[channel].tolist())

        self.app.processEvents()


class Graph2:
    def __init__(self, board2):
        self.board_id2 = board2.get_board_id()
        self.board_shim2 = board2
        self.eeg_channels2 = BoardShim.get_eeg_channels(self.board_id2)
        self.sampling_rate2 = BoardShim.get_sampling_rate(self.board_id2)
        self.update_speed_ms2 = 50
        self.window_size2 = 4
        self.num_points2 = self.window_size2 * self.sampling_rate2

        self.app2 = QtWidgets.QApplication([])
        self.win2 = pg.GraphicsLayoutWidget(show=True, title="BOARD 2")


        self._init_timeseries2()
        self._init_proccesed2()


        timer = QtCore.QTimer()
        timer.timeout.connect(self.update2)
        timer.timeout.connect(self.preproccesing2)
        timer.start(self.update_speed_ms2)
        QtWidgets.QApplication.instance().exec_()

    def _init_timeseries2(self):
        self.plotsa = list()
        self.curvesa = list()
        for i in range(len(self.eeg_channels2)):
            a = self.win2.addPlot(row=i, col=0)
            a.showAxis('left', False)
            a.setMenuEnabled('left', False)
            a.showAxis('bottom', False)
            a.setMenuEnabled('bottom', False)
            if i == 0:
                a.setTitle('Raw Data')
            self.plotsa.append(a)
            curvea = a.plot()
            self.curvesa.append(curvea)

    def _init_proccesed2(self):
        self.plotsa2 = list()
        self.curvesa2 = list()
        for i in range(len(self.eeg_channels2)):
            a2 = self.win2.addPlot(row=i, col=1)
            a2.showAxis('left', False)
            a2.setMenuEnabled('left', False)
            a2.showAxis('bottom', False)
            a2.setMenuEnabled('bottom', False)
            if i == 0:
                a2.setTitle('Proccesed Signal')
            self.plotsa2.append(a2)
            curvea2 = a2.plot()
            self.curvesa2.append(curvea2) 

    def update2(self):
        data3 = self.board_shim2.get_current_board_data(self.num_points2)
        for count, channel in enumerate(self.eeg_channels2):
            # plot timeseries
            self.curvesa[count].setData(data3[channel].tolist())

        self.app2.processEvents()


    def preproccesing2(self):
        data4 = self.board_shim2.get_current_board_data(self.num_points2)
        for count, channel in enumerate(self.eeg_channels2):
            # plot timeseries
            DataFilter.detrend(data4[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data4[channel], self.sampling_rate2, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data4[channel], self.sampling_rate2, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data4[channel], self.sampling_rate2, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curvesa2[count].setData(data4[channel].tolist())

        self.app2.processEvents()

######first try to put a graphic of bispectrum on real time###############
class Graph3:
    def __init__(self, df_gamma_average):
        #Data parameters to establish connections
        self.board_shim = df_gamma_average
        self.sampling_rate = 1
        self.update_speed_ms = 50
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
        p = self.win.addPlot(row=0, col=0)
        p.showAxis('left', False)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', False)
        p.setMenuEnabled('bottom', False)
        self.plots.append(p)
        curve = p.plot()
        self.curves.append(curve)

    def update(self):
        data = self.board_shim
        self.curves.setData(data.tolist())

        self.app.processEvents()

