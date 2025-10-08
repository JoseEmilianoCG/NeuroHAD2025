import os
import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, LogLevels
from brainflow.data_filter import DataFilter, WindowOperations
from neuro_dashboard import push
from sklearn.preprocessing import MinMaxScaler
import itertools
import math

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

def sigmoid_scale_df(result_df, ref_df, k=1.5):
    scaled_df = pd.DataFrame(index=result_df.index)
    for col in result_df.columns:
        mean = ref_df[col].mean()
        std = ref_df[col].std()
        z = (result_df[col] - mean) / (std + 1e-8)
        scaled_df[col] = 1 / (1 + np.exp(-k * z))
    return scaled_df



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
        metricscols = ['Asym1','Asym2','Att1','Att2','Rel1','Rel2','Act1','Act2','Inv1','Inv2']
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

            bands = {
                'Delta': (0, delta_limit),
                'Theta': (delta_limit, theta_limit),
                'Alpha': (theta_limit, alpha_limit),
                'Beta':  (alpha_limit, beta_limit),
                'Gamma': (beta_limit, gamma_limit)
            }


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


                cols = ['AF7-AF7', 'AF8-AF8']



                # Diccionario para guardar las medias
                freqs_dict = {}

                for band, (start, end) in bands.items():
                    # Promedio de cada canal en la banda
                    means = df_bispec[cols].iloc[start:end].mean(axis=0)
                    # Renombrar cada columna como "Banda-Canal"
                    for ch in means.index:
                        freqs_dict[f"{band}-{ch}"] = means[ch]

                # Convertir a DataFrame con una sola fila
                freqs = pd.DataFrame([freqs_dict])
                

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
                    metrics_ref_df = pd.DataFrame(refmat_arr, columns=metricscols)


                    for n in range(nwind):
                        metrics_ref_df.loc[n,'Asym1'] = alphaasymmetry(eeg_ref_df1.iloc[n*nfft:(n+1)*nfft,1], eeg_ref_df1.iloc[n*nfft:(n+1)*nfft,2], 256, nfft)
                        metrics_ref_df.loc[n,'Asym2'] = alphaasymmetry(eeg_ref_df2.iloc[n*nfft:(n+1)*nfft,1], eeg_ref_df2.iloc[n*nfft:(n+1)*nfft,2], 256, nfft)
                        metrics_ref_df.loc[n, ['Att1','Rel1','Act1','Inv1']] = multimetric(
                            eeg_ref_df1.iloc[n*nfft:(n+1)*nfft], 256, nfft
                        )
                        metrics_ref_df.loc[n, ['Att2','Rel2','Act2','Inv2']] = multimetric(
                            eeg_ref_df2.iloc[n*nfft:(n+1)*nfft], 256, nfft
                        )
                    # scalers = {}
                    # for col in cols:
                    #     print(col)
                    #     # trimmed = trim_percentiles(refmat_df[col], 0.5, 95).to_numpy().reshape(-1, 1)
                    #     scaler = MinMaxScaler(feature_range=(0,1), clip=True,)
                    #     x = np.asarray(refmat_df[[col]], dtype=float)
                    #     x = x[np.isfinite(x)]
                    #     scaler.fit(x.reshape(-1,1))
                    #     scalers[col] = scaler


                    ref_taken = True

                path_cal = f"{folder}/Calibration_data.csv"

            
            result_dict = {}
            cols = ['AF7-AF7', 'AF8-AF8']

            for band, (start, end) in bands.items():
                # Calcular promedio por canal en ese rango
                means = df_bispec[cols].iloc[start:end].mean(axis=0)
                # Renombrar cada valor con formato "Banda-Canal"
                for ch in means.index:
                    result_dict[f"{band}-{ch}"] = means[ch]

            # Convertir todo a un DataFrame con una sola fila
            result_df = pd.DataFrame([result_dict])
            print(result_df)


            # Realtime plot
            if ref_taken:
                bands_order = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

                scaled_bispec_df = sigmoid_scale_df(result_df, bispectrum_ref_df, k=1.5)

                # Promediar por banda
                band_means = {}
                for band in bands_order:
                    band_cols = [c for c in scaled_bispec_df.columns if c.startswith(f"{band}-")]
                    if band_cols:
                        band_means[band] = float(scaled_bispec_df[band_cols].mean(axis=1).iloc[0])


                # 3) DataFrame final con columnas en el orden deseado
                final_df = pd.DataFrame([band_means], columns=bands_order)


                asym1 = alphaasymmetry(procdata1[1], procdata1[2], 256, nfft)
                asym2 = alphaasymmetry(procdata2[1], procdata2[2], 256, nfft)
                att1, rel1, act1, inv1 = multimetric(procdata1, 256, nfft)
                att2, rel2, act2, inv2 = multimetric(procdata2, 256, nfft)

                metrics_result_df = pd.DataFrame({
                    'Asym1': [asym1],
                    'Asym2': [asym2],
                    'Att1': [att1],
                    'Att2': [att2],
                    'Rel1': [rel1],
                    'Rel2': [rel2],
                    'Act1': [act1],
                    'Act2': [act2],
                    'Inv1': [inv1],
                    'Inv2': [inv2]
                })
                scaled_metrics_df = sigmoid_scale_df(metrics_result_df, metrics_ref_df, k=1.5)


                # asym1s = scalers['Asym1'].transform([[asym1]])[0][0]
                # asym2s = scalers['Asym2'].transform([[asym2]])[0][0]
                # att1s = scalers['Att1'].transform([[att1]])[0][0]
                # att2s = scalers['Att2'].transform([[att2]])[0][0]
                # rel1s = scalers['Rel1'].transform([[rel1]])[0][0]
                # rel2s = scalers['Rel2'].transform([[rel2]])[0][0]
                # act1s = scalers['Act1'].transform([[act1]])[0][0]
                # act2s = scalers['Act1'].transform([[act2]])[0][0]
                # inv1s = scalers['Inv1'].transform([[inv1]])[0][0]
                # inv2s = scalers['Inv2'].transform([[inv2]])[0][0]


                radardict = {
                    "Participante 1": {
                        "Atención": scaled_metrics_df['Att1'],
                        "Relajación": scaled_metrics_df['Rel1'],
                        "Activación": scaled_metrics_df['Act1'],
                        "Emoción Positiva": 1- scaled_metrics_df['Asym1'],
                        "Involucramiento":scaled_metrics_df['Inv1']
                    },
                    "Participante 2": {
                        "Atención": scaled_metrics_df['Att2'],
                        "Relajación": scaled_metrics_df['Rel2'],
                        "Activación": scaled_metrics_df['Act2'],
                        "Emoción Positiva": 1- scaled_metrics_df['Asym2'],
                        "Involucramiento":scaled_metrics_df['Inv2']
                    },
                }
                series = {
                    "Delta": final_df['Delta'],
                    "Theta": final_df['Theta'],
                    "Alpha": final_df['Alpha'],
                    "Beta": final_df['Beta'],
                    "Gamma": final_df['Gamma'],
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
