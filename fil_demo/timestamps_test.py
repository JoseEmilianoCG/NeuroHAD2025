import os
import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

folder = "out"  # carpeta destino
os.makedirs(folder, exist_ok=True)
csv_path = os.path.join(folder, "muse_eeg_with_ts.csv")

params = BrainFlowInputParams()
# Usa uno de estos dos (según cómo conectes tu Muse 2):
params.serial_number = 'Muse-023B'     # <-- cambia por el tuyo
# params.mac_address = "XX:XX:XX:XX:XX:XX"

board_id = BoardIds.MUSE_2_BOARD.value
sr = BoardShim.get_sampling_rate(board_id)
window = 4 * sr  # 4 segundos
eeg_chs = BoardShim.get_eeg_channels(board_id)
ts_ch = BoardShim.get_timestamp_channel(board_id)
names = BoardShim.get_board_descr(board_id)["eeg_names"].split(",")  # ej. ["TP9","AF7","AF8","TP10"]

BoardShim.enable_dev_board_logger()
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream(45000)

try:
    for _ in range(3):  # 3 ventanas de ejemplo
        time.sleep(4)
        data = board.get_current_board_data(window)

        # timestamps
        board_ts = data[ts_ch]
        t_pull = time.time()
        unix_ts = t_pull - (len(board_ts)-1)/sr + np.arange(len(board_ts))/sr

        # armar DataFrame: 4 canales + timestamps
        df = pd.DataFrame({names[i]: data[ch] for i, ch in enumerate(eeg_chs)})
        df["board_ts"] = board_ts
        df["unix_ts"] = unix_ts

        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

finally:
    board.stop_stream()
    board.release_session()

print(f"Guardado en: {csv_path}")
