import os
import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, LogLevels
from brainflow.data_filter import DataFilter, WindowOperations
from neuro_dashboard import push
from sklearn.preprocessing import MinMaxScaler
import itertools

def bandpower(data, fs, fmin, fmax, nfft, ovlap):
    # print(fs, fmin, fmax, nfft, ovlap)
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64).reshape(-1))
    psd = DataFilter.get_psd_welch(x, nfft, int(nfft*ovlap), fs, WindowOperations.HANNING)
    power = DataFilter.get_band_power(psd, fmin, fmax)

    return power


def alphaasymmetry(
    datach1,
    datach2,
    fs,
    nfft
):
    # Potencia integrada alpha
    alpha1 = bandpower(datach1, fs, 8.0, 12.0, nfft, 0.5)
    alpha2 = bandpower(datach2, fs, 8.0, 12.0, nfft, 0.5)

    alpha1 = np.where(alpha1 > 0.0000000001, alpha1, 1e-9)
    alpha2 = np.where(alpha2 > 0.0000000001, alpha2, 1e-9)

    lnalpha1 = np.log(alpha1, where=alpha1 > 0)
    lnalpha2 = np.log(alpha2, where=alpha2 > 0)

    asymm = lnalpha2 - lnalpha1
    # score = 1.0 / (1.0 + np.exp(4.0 * asymm))

    return asymm


def multimetric(
    data,
    fs,
    nfft
):
    att = np.mean([bandpower(data[ch], fs, 4.0, 8.0, nfft, 0.5) for ch in range(4)])
    rel = np.mean([bandpower(data[ch], fs, 8.0, 13.0, nfft, 0.5) for ch in range(4)])
    act = np.mean([bandpower(data[ch], fs, 13.0, 30.0, nfft, 0.5) for ch in range(4)])
    inv = act / (att + rel + 1e-9)

    return att, rel, act, inv

def trim_percentiles(series, low=0.5, high=95):
    q_low, q_high = np.percentile(series, [low, high])
    return series[(series >= q_low) & (series <= q_high)]


# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
def bispec(
    eno1_datach1,
    eno1_datach2,
    eno1_datach3,
    eno1_datach4,
    eno2_datach1,
    eno2_datach2,
    eno2_datach3,
    eno2_datach4,
    second,
    basaltime,
    totaltime,
    sleeptime,
    folder,
    dash_q,
):
    try:
        eeg_ref_list1 = []
        eeg_ref_list2 = []
        bispectrum_ref_list = []
        ref_taken = False
        nfft = 256 * sleeptime
        channels = ['TP9','AF7','AF8','TP10']
        pairs = list(itertools.product(channels, repeat=2))
        pairs = [x[0] + '-' + x[1] for x in pairs]
        while True:
            time.sleep(sleeptime)
            df_bispecMV1 = eno1_datach1
            df_bispecMV2 = eno1_datach2
            df_bispecMV3 = eno1_datach3
            df_bispecMV4 = eno1_datach4

            procdata1 = np.array(
                [
                    df_bispecMV1[0:nfft],
                    df_bispecMV2[0:nfft],
                    df_bispecMV3[0:nfft],
                    df_bispecMV4[0:nfft],
                ]
            )

            df_bispec2MV1 = eno2_datach1
            df_bispec2MV2 = eno2_datach2
            df_bispec2MV3 = eno2_datach3
            df_bispec2MV4 = eno2_datach4

            procdata2 = np.array(
                [
                    df_bispec2MV1[0:nfft],
                    df_bispec2MV2[0:nfft],
                    df_bispec2MV3[0:nfft],
                    df_bispec2MV4[0:nfft],
                ]
            )

            matrix_eno1 = np.array(
                [df_bispecMV1[:], df_bispecMV2[:], df_bispecMV3[:], df_bispecMV4[:]]
            )
            matrix_eno1t = matrix_eno1.transpose()
            matrix_eno2 = np.array(
                [df_bispec2MV1[:], df_bispec2MV2[:], df_bispec2MV3[:], df_bispec2MV4[:]]
            )
            matrix_eno2t = matrix_eno2.transpose()

            cont = 0
            Nch = 4

            B = np.zeros((Nch * Nch, len(df_bispecMV1) // 2))
            index = np.zeros((Nch * Nch, 2))

            for ch2 in range(Nch):
                for ch1 in range(Nch):
                    bs = np.abs(
                        np.fft.fft(matrix_eno1t[:, ch1])
                        * np.fft.fft(matrix_eno2t[:, ch2])
                        * np.conj(
                            np.fft.fft(matrix_eno1t[:, ch1] + matrix_eno2t[:, ch2])
                        )
                    )
                    bs_t = bs[: len(bs) // 2].T
                    result = np.where(bs_t > 0.0000000001, bs_t, 1e-9)
                    B[cont, :] = np.log(
                        result, where=result > 0
                    )  # Mean windows bs on all channels
                    index[cont, :] = [
                        ch1 + 1,
                        ch2 + 1,
                    ]  # Indexing combination order: ch1,ch2
                    cont += 1
                ## Revisar
                # df_time[Nch] = B[Nch]
            print(B)

            bispectrum = pd.DataFrame(B)
            b_transpose = bispectrum.transpose()

            df_bispec = pd.DataFrame(
                columns=[pairs]
            )
            for eeg_channel2 in range(0, 16):
                df_bispec[pairs[eeg_channel2]] = b_transpose[eeg_channel2]
            df_norm = np.zeros((len(df_bispec), Nch * Nch))
            # print(df_bispec)
            # df_norm = pd.DataFrame()
            df_bispec.to_csv("{}/Bispec.csv".format(folder), mode="a")
            # df_norm = pd.DataFrame()

            # State frequency band limits
            delta_limit = (
                (4 * len(df_bispec)) // 128
            )  # 125Hz is the frequency limit to the bispectrum matrix length data
            theta_limit = (8 * len(df_bispec)) // 128
            alpha_limit = (13 * len(df_bispec)) // 128
            beta_limit = (29 * len(df_bispec)) // 128
            gamma_limit = (50 * len(df_bispec)) // 128


            cal_written = False
            plot = False

            with second.get_lock():
                sec = int(second.value)

            print(f"[bispec] sec={sec}", flush=True)

            # 2) condiciones SIN lock
            if sec >= totaltime:
                break

            if 2 < sec <= basaltime and not cal_written:
                print("Preparing device calibration...", flush=True)
                path_cal = f"{folder}/Calibration_data.csv"
                os.makedirs(folder, exist_ok=True)
                df_bispec.to_csv(
                    path_cal, mode="a", header=not os.path.exists(path_cal), index=False
                )
                cal_written = True

                eeg_ref_list1.append(pd.DataFrame(procdata1.T))
                eeg_ref_list2.append(pd.DataFrame(procdata2.T))


                delta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[0:delta_limit, :].mean(axis=0).mean()
                theta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[delta_limit:theta_limit, :].mean(axis=0).mean()
                alpha = df_bispec[['AF7-AF7','AF8-AF8']].iloc[theta_limit:alpha_limit, :].mean(axis=0).mean()
                beta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[alpha_limit:beta_limit, :].mean(axis=0).mean()
                gamma = df_bispec[['AF7-AF7','AF8-AF8']].iloc[beta_limit:gamma_limit, :].mean(axis=0).mean()
                freqs = pd.DataFrame({
                    'Delta': [float(delta)],
                    'Theta': [float(theta)],
                    'Alpha': [float(alpha)],
                    'Beta':  [float(beta)],
                    'Gamma': [float(gamma)]
                })
                

                bispectrum_ref_list.append(freqs)






            elif basaltime < sec <= totaltime:
                if not ref_taken:
                    eeg_ref_df1 = pd.concat(eeg_ref_list1, ignore_index=True)
                    eeg_ref_df2 = pd.concat(eeg_ref_list2, ignore_index=True)
                    bispectrum_ref_df = pd.concat(bispectrum_ref_list, ignore_index=True)
                    print(bispectrum_ref_df.columns)

                    nwind = eeg_ref_df1.shape[0] // nfft
                    # print('Refmat windows: ', nwind)
                    refmat_arr = np.zeros([nwind,10])
                    cols = ['Asym1','Asym2','Att1','Att2','Rel1','Rel2','Act1','Act2','Inv1','Inv2']
                    refmat_df = pd.DataFrame(refmat_arr, columns=cols)


                    for n in range(nwind):
                        refmat_df.loc[n,'Asym1'] = alphaasymmetry(eeg_ref_df1.iloc[n*nfft:(n+1)*nfft,1], eeg_ref_df1.iloc[n*nfft:(n+1)*nfft,2], 256, nfft)
                        refmat_df.loc[n,'Asym2'] = alphaasymmetry(eeg_ref_df2.iloc[n*nfft:(n+1)*nfft,1], eeg_ref_df2.iloc[n*nfft:(n+1)*nfft,2], 256, nfft)
                        refmat_df.loc[n, ['Att1','Rel1','Act1','Inv1']] = multimetric(
                            eeg_ref_df1.iloc[n*nfft:(n+1)*nfft], 256, nfft
                        )
                        refmat_df.loc[n, ['Att2','Rel2','Act2','Inv2']] = multimetric(
                            eeg_ref_df2.iloc[n*nfft:(n+1)*nfft], 256, nfft
                        )
                    scalers = {}
                    for col in cols:
                        print(col)
                        # trimmed = trim_percentiles(refmat_df[col], 0.5, 95).to_numpy().reshape(-1, 1)
                        scaler = MinMaxScaler(feature_range=(0,1), clip=True,)
                        x = np.asarray(refmat_df[[col]], dtype=float)
                        x = x[np.isfinite(x)]
                        scaler.fit(x.reshape(-1,1))
                        scalers[col] = scaler

                    bispecscaler = {}
                    for col in bispectrum_ref_df.columns:
                        scaler = MinMaxScaler(feature_range=(0,1), clip=True)
                        x = np.asarray(bispectrum_ref_df[[col]], dtype=float)
                        x = x[np.isfinite(x)]
                        trimmed = trim_percentiles(x, 0.5, 95).reshape(-1, 1)
                        scaler.fit(trimmed)
                        bispecscaler[col] = scaler

                    ref_taken = True

                path_cal = f"{folder}/Calibration_data.csv"
            #     if os.path.exists(path_cal):
            #         sumdf = pd.read_csv(path_cal)
            #         sumdf.drop('Time', axis=1)
            #         arrange3 = (
            #             sumdf.apply(pd.to_numeric, errors="coerce")
            #             .dropna(axis=0)
            #             .reset_index(drop=True)
            #         )
            #         arrange3.to_csv(f"{folder}/Calibration_data_clean.csv", index=False)

            #         eyes_open = pd.read_csv(
            #             "{}/Calibration_data_clean.csv".format(folder)
            #         )

            #         df_eo = (
            #             eyes_open.apply(pd.to_numeric, errors="coerce")
            #             .dropna()
            #             .reset_index(drop=True)
            #         )

            #         df_eo2 = df_eo.rename(columns={f"COMB{k}": k for k in range(16)})


            #         relevant_keys = np.arange(0, len(df_eo), nfft/2)
            #         df_norm = np.zeros((nfft/2, 16), dtype=float)

            #         for i in range(nfft/2):
            #             idx = relevant_keys + i
            #             idx = idx[idx < len(df_eo2)]  # límites
            #             vals = df_eo2.to_numpy()[idx, :]  # (n_idx x 16)
            #             df_norm[i, :] = vals.mean(axis=0)  # promedio por combinación


            # df_sum = pd.DataFrame(df_norm)

            # df_sum2 = df_sum.rename(
            #     columns={
            #         0: "COMB0",
            #         1: "COMB1",
            #         2: "COMB2",
            #         3: "COMB3",
            #         4: "COMB4",
            #         5: "COMB5",
            #         6: "COMB6",
            #         7: "COMB7",
            #         8: "COMB8",
            #         9: "COMB9",
            #         10: "COMB10",
            #         11: "COMB11",
            #         12: "COMB12",
            #         13: "COMB13",
            #         14: "COMB14",
            #         15: "COMB15",
            #     }
            # )
            # df_sub = df_bispec.sub(df_sum2)
            # df_div = df_sub.div(df_sum2)
            # print(df_div)

            # df_div.to_csv("{}/Bispec_norm.csv".format(folder), mode="a")


            df_delta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[0:delta_limit, :].mean(axis=0).mean()
            df_theta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[delta_limit:theta_limit, :].mean(axis=0).mean()
            df_alpha = df_bispec[['AF7-AF7','AF8-AF8']].iloc[theta_limit:alpha_limit, :].mean(axis=0).mean()
            df_beta = df_bispec[['AF7-AF7','AF8-AF8']].iloc[alpha_limit:beta_limit, :].mean(axis=0).mean()
            df_gamma = df_bispec[['AF7-AF7','AF8-AF8']].iloc[beta_limit:gamma_limit, :].mean(axis=0).mean()
            print(df_gamma)

            result_df = pd.DataFrame({
                    'Delta': [float(df_delta)],
                    'Theta': [float(df_theta)],
                    'Alpha': [float(df_alpha)],
                    'Beta':  [float(df_beta)],
                    'Gamma': [float(df_gamma)]
                })

            # Concatenate the individual DataFrames horizontally (column-wise)
            # result_df = pd.concat(
            #     [df_delta, df_theta, df_alpha, df_beta, df_gamma], axis=0
            # )

            # Realtime plot
            if ref_taken:
                comb1s = bispecscaler['Delta'].transform([[df_delta]])
                comb2s = bispecscaler['Theta'].transform([[df_theta]])
                comb3s = bispecscaler['Alpha'].transform([[df_alpha]])
                comb4s = bispecscaler['Beta'].transform([[df_beta]])
                comb5s = bispecscaler['Gamma'].transform([[df_gamma]])


                asym1 = alphaasymmetry(procdata1[1], procdata1[2], 256, nfft)
                asym2 = alphaasymmetry(procdata2[1], procdata2[2], 256, nfft)
                att1, rel1, act1, inv1 = multimetric(procdata1, 256, nfft)
                att2, rel2, act2, inv2 = multimetric(procdata2, 256, nfft)

                asym1s = scalers['Asym1'].transform([[asym1]])[0][0]
                asym2s = scalers['Asym2'].transform([[asym2]])[0][0]
                att1s = scalers['Att1'].transform([[att1]])[0][0]
                att2s = scalers['Att2'].transform([[att2]])[0][0]
                rel1s = scalers['Rel1'].transform([[rel1]])[0][0]
                rel2s = scalers['Rel2'].transform([[rel2]])[0][0]
                act1s = scalers['Act1'].transform([[act1]])[0][0]
                act2s = scalers['Act1'].transform([[act2]])[0][0]
                inv1s = scalers['Inv1'].transform([[inv1]])[0][0]
                inv2s = scalers['Inv2'].transform([[inv2]])[0][0]


                radardict = {
                    "Participante 1": {
                        "Atención": att1s,
                        "Relajación": rel1s,
                        "Activación": act1s,
                        "Emoción Positiva": 1- asym1s,
                        "Involucramiento":inv1s
                    },
                    "Participante 2": {
                        "Atención": att2s,
                        "Relajación": rel2s,
                        "Activación": act2s,
                        "Emoción Positiva": 1 - asym2s,
                        "Involucramiento": inv2s
                    },
                }
                series = {
                    "Delta": comb1s,
                    "Theta": comb2s,
                    "Alpha": comb3s,
                    "Beta": comb4s,
                    "Gamma": comb5s,
                }
                push(
                    dash_q,
                    sec,
                    series_dict=series,
                    radars=radardict,
                )

                # Si algún día quieres mandar 2 puntos cada 4 s:
                # push(dash_q, [sec, sec+2], {"Sincronía Alfa AF7 - AF7": [v1, v2]})

            # Transpose the concatenated DataFrame to have a shape of [1 row x 20 columns]
            # bispectrum_mean = pd.DataFrame(result_df).transpose()

            # Create a list of new column names with both the combination number and frequency band
            new_column_names = []

            # Define the frequency bands
            frequency_bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
            n_combos = int(index.shape[0])  # 16
            new_column_names = [
                f"COMB{comb}_{band}"
                for band in frequency_bands
                for comb in range(n_combos)
            ]

            # Assign the new column names to the DataFrame
            # bispectrum_mean.columns = new_column_names

            # matrix = pd.DataFrame(arrange3).transpose()
            result_df.to_csv(
                "{}/Frequency_bands_bispectrum.csv".format(folder), mode="a"
            )

            print(result_df)
            # Graph3(df_gamma_average)

    except KeyboardInterrupt:
        BoardShim.log_message(
            LogLevels.LEVEL_INFO.value, " ---- End the session with Muse 2 ---"
        )
