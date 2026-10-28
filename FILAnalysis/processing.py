import numpy as np
import pandas as pd
from itertools import product
import mne
import mne_connectivity as mnec
from sklearn.preprocessing import StandardScaler


### =====================TIMEWINDOW SEGMENTATION===================== ###
def timewindow(data=np.array([]), fs=1, winsz=1, ovlap=0):
    # Parameters and data
    wn = winsz * fs  # Samples per window
    on = int(np.floor(ovlap * wn))  # Samples in overlap
    dl = len(data)  # Length of data
    bN = int((dl - wn) / (wn - on) + 1)  # Buffer number

    print(wn, on, dl, normalize_by_baseline)

    # Buffering
    # This loop goes through rows of the output matrix and starting index of data, first range inside zip establishes rows, the second one
    # states the indexes, going from value index 0 to data length minus window size plus one, with step of window size minus overlap
    outdata = np.zeros([bN, wn])  # Output data
    for row, idx in zip(range(0, bN), range(0, (dl - wn) + 1, wn - on)):
        outdata[row, :] = data[idx : idx + wn]

    return outdata


### =====================PROCESSING FUNCTION===================== ###
def process(
    preprodata, stages, sessions, min_freq, max_freq, fs, winsz, ovlap, channels
):

    # FREQUENCY SETUP

    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))
    fmin = tuple(np.arange(min_freq, max_freq, 1))
    fmax = tuple(np.arange(min_freq, max_freq, 1) + 1)

    connectivity_methods = ["coh", "plv", "pli", "wpli"]
    n_con_methods = len(connectivity_methods)

    # CHANNEL SETUP

    participants = ["1", "2"]
    n_ch = len(channels)

    ch_names = [f"{ch}_{p}" for p in participants for ch in channels]

    idx_p1 = np.arange(n_ch)
    idx_p2 = np.arange(n_ch, 2 * n_ch)

    indices = (
        np.repeat(idx_p1, n_ch).tolist(),
        np.tile(idx_p2, n_ch).tolist(),
    )

    ch_pairs = [f"{i}_{j}" for i in channels for j in channels]
    band_names = [f"{i}-{j} Hz" for i, j in zip(fmin, fmax)]
    cols = [f"{ch}_{band}" for ch, band in product(ch_pairs, band_names)]

    info = mne.create_info(ch_names, fs, ch_types="eeg")

    conndata = {}

    # MAIN LOOP

    for stage in stages:
        session_data = {}

        for session_id in sessions:
            print(stage, session_id)
            # --- Extract data ---
            p1_data = (
                preprodata["p1"][stage][
                    preprodata["p1"][stage]["Session"] == session_id
                ]
                .drop("Session", axis=1)
                .to_numpy()
                .T
            )

            p2_data = (
                preprodata["p2"][stage][
                    preprodata["p2"][stage]["Session"] == session_id
                ]
                .drop("Session", axis=1)
                .to_numpy()
                .T
            )

            # --- Windowing (sin overhead extra) ---
            p1_holder = [timewindow(sig, fs, winsz, ovlap) for sig in p1_data]
            p2_holder = [timewindow(sig, fs, winsz, ovlap) for sig in p2_data]

            p1_arr = np.array(p1_holder)
            p2_arr = np.array(p2_holder)

            data = np.vstack((p1_arr, p2_arr))
            data = np.transpose(data, (1, 0, 2))

            data_epoch = mne.EpochsArray(data, info)

            # --- Connectivity (preallocación eficiente) ---
            con_time = mnec.spectral_connectivity_time(
                data_epoch,
                freqs,
                method=connectivity_methods,
                sfreq=fs,
                mode="cwt_morlet",
                indices=indices,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
            )

            con_time_array = np.empty(
                (n_con_methods, data.shape[0], n_ch**2, len(fmin))
            )

            for c in range(n_con_methods):
                con_time_array[c] = con_time[c].get_data(output="compact")

            session_data[session_id] = con_time_array

        conndata[stage] = session_data

    # DATAFRAME BUILDING

    procdata = {}

    for stage in stages:
        dfs = []

        for session_id in sessions:
            for metric_idx, metric_name in enumerate(connectivity_methods):
                conn_matrix = conndata[stage][session_id][metric_idx]
                conn_matrix = conn_matrix.reshape(conn_matrix.shape[0], -1)

                df_sess = pd.DataFrame(conn_matrix, columns=cols)
                df_sess.insert(0, "Session", session_id)
                df_sess.insert(1, "Metric", metric_name)
                df_sess.insert(2, "Epoch", np.arange(1, conn_matrix.shape[0] + 1))

                dfs.append(df_sess)

        procdata[stage] = pd.concat(dfs, ignore_index=True)

    return procdata


### =====================BAND AVERAGING===================== ###
def get_band_average(procdata, band_freqs, stages, channels):
    ch_comb = list(product(channels, repeat=2))

    band_procdata = {}

    for stage in stages:
        base_df = procdata[stage][["Session", "Metric", "Epoch"]].copy()

        for comb in ch_comb:
            ch1, ch2 = comb

            for band in list(band_freqs.keys()):
                cols = [
                    f"{ch1}_{ch2}_{i}-{i + 1} Hz"
                    for i in range(band_freqs[band][0], band_freqs[band][1])
                ]

                mean_holder = procdata[stage][cols].mean(axis=1)

                base_df[f"{ch1}_{ch2}_{band}"] = mean_holder

        band_procdata[stage] = base_df
    return band_procdata


### =====================NORMALIZATION===================== ###
def normalize_by_baseline(band_procdata):
    # Columns that should NOT be normalized
    id_cols = ["Session", "Metric", "Epoch"]
    feature_cols = [c for c in band_procdata["baseline"].columns if c not in id_cols]

    # Initialize output structure
    procdata_norm = {stage: [] for stage in band_procdata}

    sessions = band_procdata["baseline"]["Session"].unique()
    metrics = band_procdata["baseline"]["Metric"].unique()

    for session in sessions:
        for metric in metrics:
            # --- Fit scaler using baseline for this session and metric ---
            base_df = band_procdata["baseline"].loc[
                (band_procdata["baseline"]["Session"] == session)
                & (band_procdata["baseline"]["Metric"] == metric)
            ]

            scaler = StandardScaler().fit(base_df[feature_cols])

            # --- Apply transformation to all stages ---
            for stage in band_procdata:
                df = band_procdata[stage]

                df_sub = df.loc[
                    (df["Session"] == session) & (df["Metric"] == metric)
                ].copy()

                # Apply normalization using baseline statistics
                df_sub[feature_cols] = scaler.transform(df_sub[feature_cols])

                procdata_norm[stage].append(df_sub)

    # Concatenate all processed chunks
    procdata_norm = {
        stage: pd.concat(procdata_norm[stage], ignore_index=True)
        for stage in procdata_norm
    }

    return procdata_norm


### =====================FEATURE MATRIX===================== ###
def build_feature_matrix(dfs_by_stage, metric="pli"):
    import pandas as pd

    # Columns that are NOT features
    id_cols = ["Session", "Metric", "Epoch"]
    feature_cols = [c for c in dfs_by_stage["shared"].columns if c not in id_cols]

    # --- Filter by metric ---
    df_shared = dfs_by_stage["shared"]
    df_shared = df_shared[df_shared["Metric"] == metric].copy()

    df_individual = dfs_by_stage["individual"]
    df_individual = df_individual[df_individual["Metric"] == metric].copy()

    # --- Add target column ---
    df_shared["Target"] = 0
    df_individual["Target"] = 1

    # --- Concatenate ---
    feat_mat = pd.concat([df_shared, df_individual], ignore_index=True)
    feat_mat = feat_mat[["Session", "Target"] + feature_cols]

    return feat_mat
