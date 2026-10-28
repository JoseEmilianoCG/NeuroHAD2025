import os
import time
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations


LIVE_MODE = 0
CSV_MODE = 1


EEG_COLS = ["TP9", "AF7", "AF8", "TP10"]
TS_COLS = ["board_ts", "unix_ts"]


def pad_or_trim(seq, target_len, pad_value=0.0):
    m = len(seq)
    if m >= target_len:
        return list(seq)[-target_len:]
    return [pad_value] * (target_len - m) + list(seq)


def _sanitize_chunk_df(df, channel_names):
    """
    Convierte a numérico y elimina filas basura, por ejemplo headers repetidos
    dentro del CSV cuando se concatenaron bloques con header=True.
    """
    keep_cols = [c for c in list(channel_names[:4]) + TS_COLS if c in df.columns]
    df = df.loc[:, keep_cols].copy()

    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eeg_cols_present = [c for c in channel_names[:4] if c in df.columns]
    if eeg_cols_present:
        df = df.dropna(subset=eeg_cols_present, how="any")

    return df.reset_index(drop=True)


class CsvChunkReader:
    def __init__(self, csv_path, chunk_size, sampling_rate, channel_names):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names[:4]
        self.df = pd.read_csv(csv_path)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])

        missing = [c for c in self.channel_names if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"El CSV '{csv_path}' no contiene estas columnas requeridas: {missing}"
            )

        self.df = _sanitize_chunk_df(self.df, self.channel_names)
        self.idx = 0

    def next_chunk(self):
        ini = self.idx
        fin = self.idx + self.chunk_size
        chunk = self.df.iloc[ini:fin].copy()
        if chunk.empty:
            return None

        if "board_ts" not in chunk.columns:
            n = len(chunk)
            start_ts = self.idx / self.sampling_rate
            chunk["board_ts"] = start_ts + np.arange(n) / self.sampling_rate

        if "unix_ts" not in chunk.columns:
            n = len(chunk)
            base = time.time()
            chunk["unix_ts"] = base + np.arange(n) / self.sampling_rate

        self.idx = fin
        return chunk


class LiveMuseReader:
    def __init__(self, muse_id, folder, file_suffix=""):
        self.muse_id = muse_id
        self.folder = folder
        self.file_suffix = file_suffix
        self.board = None
        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)[:4]
        self.channel_names = BoardShim.get_board_descr(self.board_id)["eeg_names"].split(",")
        self.ts_ch = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self._started = False

    def start(self):
        if self._started:
            return
        print(self.muse_id)
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_number = self.muse_id
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        out_name = f"testOpenBCI{self.file_suffix}.csv"
        self.board.start_stream(45000, f"file://{self.folder}/{out_name}:w")
        BoardShim.log_message(
            LogLevels.LEVEL_INFO.value,
            f" ---- Starting the streaming with {self.muse_id} ---",
        )
        self._started = True

    def stop(self):
        if not self._started:
            return
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass
        self._started = False
        BoardShim.log_message(
            LogLevels.LEVEL_INFO.value,
            f" ---- End the session with {self.muse_id} ---",
        )

    def next_chunk(self):
        data = self.board.get_board_data()
        if data is None or data.size == 0:
            return pd.DataFrame(columns=self.channel_names + ["board_ts", "unix_ts"]), \
                   pd.DataFrame(columns=self.channel_names + ["board_ts", "unix_ts"])

        t_pull = time.time()
        board_ts = data[self.ts_ch]
        unix_ts = (
            t_pull - (len(board_ts) - 1) / self.sampling_rate + np.arange(len(board_ts)) / self.sampling_rate
        )

        df_raw = pd.DataFrame({"board_ts": board_ts, "unix_ts": unix_ts})
        df_clean = pd.DataFrame({"board_ts": board_ts, "unix_ts": unix_ts})

        for eeg_channel in self.eeg_channels:
            values = data[eeg_channel].copy()
            col_name = self.channel_names[eeg_channel - 1]
            df_raw[col_name] = values

            DataFilter.detrend(values, DetrendOperations.LINEAR.value)
            DataFilter.remove_environmental_noise(values, self.sampling_rate, noise_type=1)
            DataFilter.perform_lowpass(
                values,
                self.sampling_rate,
                cutoff=100,
                order=4,
                filter_type=0,
                ripple=0,
            )
            DataFilter.perform_highpass(
                values,
                self.sampling_rate,
                cutoff=0.1,
                order=4,
                filter_type=0,
                ripple=0,
            )
            df_clean[col_name] = values

        return df_raw, df_clean


def _append_csv(df, path):
    if df is None or len(df) == 0:
        return
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _publish_clean_df(df_clean, datach1, datach2, datach3, datach4, channel_names):
    df_clean = _sanitize_chunk_df(df_clean, channel_names)

    info1 = df_clean[channel_names[0]].astype(float).tolist() if channel_names[0] in df_clean else []
    info2 = df_clean[channel_names[1]].astype(float).tolist() if channel_names[1] in df_clean else []
    info3 = df_clean[channel_names[2]].astype(float).tolist() if channel_names[2] in df_clean else []
    info4 = df_clean[channel_names[3]].astype(float).tolist() if channel_names[3] in df_clean else []

    window = len(datach1)
    datach1[:] = pad_or_trim(info1, window)
    datach2[:] = pad_or_trim(info2, window)
    datach3[:] = pad_or_trim(info3, window)
    datach4[:] = pad_or_trim(info4, window)


def device_provider(
    second,
    folder,
    datach1,
    datach2,
    datach3,
    datach4,
    totaltime,
    muse_id,
    source_mode,
    csv_path,
    prepro_subfolder="Prepro",
    raw_filename="Raw.csv",
    clean_filename="Clean.csv",
    chunk_sec=2,
):
    live_reader = LiveMuseReader(
        muse_id=muse_id,
        folder=folder,
        file_suffix="" if prepro_subfolder == "Prepro" else "2",
    )
    live_reader.start()

    chunk_size = int(live_reader.sampling_rate * chunk_sec)
    csv_reader = CsvChunkReader(
        csv_path=csv_path,
        chunk_size=chunk_size,
        sampling_rate=live_reader.sampling_rate,
        channel_names=live_reader.channel_names,
    )

    channel_names = live_reader.channel_names[:4]
    raw_path = os.path.join(folder, prepro_subfolder, raw_filename)
    clean_path = os.path.join(folder, prepro_subfolder, clean_filename)
    switched_to_csv = False

    try:
        while True:
            with second.get_lock():
                if second.value >= totaltime:
                    break

            with source_mode.get_lock():
                mode = source_mode.value

            if mode == CSV_MODE:
                if not switched_to_csv:
                    switched_to_csv = True
                    print(f"[{muse_id}] Cambio detectado: LIVE -> CSV ({csv_path})")
                    live_reader.stop()

                chunk = csv_reader.next_chunk()
                if chunk is None:
                    print(f"[{muse_id}] Se terminó el CSV, se mantiene la última ventana publicada.")
                    time.sleep(chunk_sec)
                    continue

                df_raw = chunk.copy()
                df_clean = chunk.copy()
                _append_csv(df_raw, raw_path)
                _append_csv(df_clean, clean_path)
                _publish_clean_df(df_clean, datach1, datach2, datach3, datach4, channel_names)
                time.sleep(chunk_sec)
            else:
                time.sleep(chunk_sec)
                df_raw, df_clean = live_reader.next_chunk()
                _append_csv(df_raw, raw_path)
                _append_csv(df_clean, clean_path)
                _publish_clean_df(df_clean, datach1, datach2, datach3, datach4, channel_names)
    except KeyboardInterrupt:
        pass
    finally:
        live_reader.stop()
