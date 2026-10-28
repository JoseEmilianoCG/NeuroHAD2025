from bispectrum import bispec
from timer import timer
from markers import run_with_window
from neuro_dashboard import run_dash, shutdown
from csv_switch_provider import device_provider, LIVE_MODE, CSV_MODE

import multiprocessing
from multiprocessing import Queue, Process, Value, Manager
import threading

import numpy as np
from colorama import Fore, Style
import scipy.stats as stats

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BoardIds


seconds = Value("i", 0)
counts = Value("i", 0)

sleeptime = 1
sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
windsz = int(sleeptime * sampling_rate)

labels = {
    "1": "baseline_start",
    "2": "shared_start",
    "3": "individual_start",
    " ": "event",
}


def keyboard_listener(source_mode):
    print("\nControles de captura:")
    print("  c + Enter  -> cambiar una sola vez de LIVE a CSV")
    print("  q + Enter  -> no hace nada especial; para terminar usa el tiempo total")
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            return
        if cmd == "c":
            with source_mode.get_lock():
                if source_mode.value == LIVE_MODE:
                    source_mode.value = CSV_MODE
                    print(Fore.YELLOW + "Cambio a modo CSV solicitado." + Style.RESET_ALL)
                else:
                    print("Ya estás en modo CSV.")


if __name__ == "__main__":
    mgr = Manager()
    eno1_datach1 = multiprocessing.Array("d", windsz)
    eno1_datach2 = multiprocessing.Array("d", windsz)
    eno1_datach3 = multiprocessing.Array("d", windsz)
    eno1_datach4 = multiprocessing.Array("d", windsz)

    eno2_datach1 = multiprocessing.Array("d", windsz)
    eno2_datach2 = multiprocessing.Array("d", windsz)
    eno2_datach3 = multiprocessing.Array("d", windsz)
    eno2_datach4 = multiprocessing.Array("d", windsz)

    dash_q = Queue()
    source_mode = Value("i", LIVE_MODE)

    session_num, repetition_num = input(
        "Introduce el número de sesión y repetición, separa con un espacio: "
    ).split(" ")
    session_num = f"{int(session_num):02d}"
    repetition_num = f"{int(repetition_num):02d}"
    basaltime = int(input("Introduce el tiempo de baseline en segundos: "))
    totaltime = int(input("Introduce el tiempo total de la sesión en segundos: "))

    print("\n=== Información adicional del experimento ===")
    muse_code1 = (
        "Muse-"
        + str(
            input(
                "ID de Muse #1, a ser usado por el adulto (preferentemennte, 023B o 06D3): "
            )
        ).upper()
    )
    muse_code2 = (
        "Muse-"
        + str(
            input(
                "ID de Muse #2, a ser usado por el infante (preferentemennte, 070E o E215): "
            )
        ).upper()
    )
    relationship = input(
        "Relación entre los sujetos (e.g. Padre-hijo, Madre-hijo, Amigos, Desconocidos, Pareja, etc.): "
    )

    # csv_path1 = input("Ruta del CSV para simular dispositivo 1: ").strip()
    # csv_path2 = input("Ruta del CSV para simular dispositivo 2: ").strip()

    csv_path1 = r'C:\Users\josee\OneDrive\Documentos\GitHub\NeuroHAD2025\B2BProto_V2\simul\Raw.csv'
    csv_path2 = r'C:\Users\josee\OneDrive\Documentos\GitHub\NeuroHAD2025\B2BProto_V2\simul\Raw2.csv'

    metadata_df = pd.DataFrame(
        [
            {
                "session_num": session_num,
                "repetition_num": repetition_num,
                "muse_code1": muse_code1,
                "muse_code2": muse_code2,
                "relationship": relationship,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "csv_path1": csv_path1,
                "csv_path2": csv_path2,
            }
        ]
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    folder_name = f"S{session_num}R{repetition_num}_{datetime.now().strftime('%d%m%Y_%H%M')}"
    folder = os.path.join(OUTPUTS_DIR, folder_name)
    os.makedirs(folder, exist_ok=True)

    for subfolder in ["Prepro", "Processed", "Figures"]:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    for subfolder2 in ["Prepro 2", "Processed 2", "Figures 2"]:
        os.makedirs(os.path.join(folder, subfolder2), exist_ok=True)

    metadata_df.to_csv(os.path.join(folder, "session_info.csv"), index=False)
    print(
        Fore.GREEN
        + "Archivo 'session_info.csv' guardado con los metadatos de la sesión."
        + Style.RESET_ALL
    )

    timestamps = mgr.list()

    threading.Thread(target=keyboard_listener, args=(source_mode,), daemon=True).start()

    p_dash = Process(
        target=run_dash,
        args=(dash_q,),
        kwargs=dict(
            port=8051,
            window_sec=60,
            xmin=30,
            title="Impacto de la Lectura Compartida en Sincronización Biométrica y Conexión Emocional",
        ),
        daemon=True,
    )

    p_dash.start()
    process2 = Process(target=timer, args=[seconds, counts, timestamps, totaltime])
    q = Process(
        target=device_provider,
        args=[
            seconds,
            folder,
            eno1_datach1,
            eno1_datach2,
            eno1_datach3,
            eno1_datach4,
            totaltime,
            muse_code1,
            source_mode,
            csv_path1,
        ],
        kwargs=dict(
            prepro_subfolder="Prepro",
            raw_filename="Raw.csv",
            clean_filename="Clean.csv",
            chunk_sec=sleeptime,
        ),
    )
    q2 = Process(
        target=device_provider,
        args=[
            seconds,
            folder,
            eno2_datach1,
            eno2_datach2,
            eno2_datach3,
            eno2_datach4,
            totaltime,
            muse_code2,
            source_mode,
            csv_path2,
        ],
        kwargs=dict(
            prepro_subfolder="Prepro 2",
            raw_filename="Raw2.csv",
            clean_filename="Clean2.csv",
            chunk_sec=sleeptime,
        ),
    )
    q3 = Process(
        target=bispec,
        args=[
            eno1_datach1,
            eno1_datach2,
            eno1_datach3,
            eno1_datach4,
            eno2_datach1,
            eno2_datach2,
            eno2_datach3,
            eno2_datach4,
            seconds,
            basaltime,
            totaltime,
            sleeptime,
            folder,
            dash_q,
        ],
    )
    q4 = Process(target=run_with_window, args=(f"{folder}/markers.csv", labels), daemon=True)

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
        shutdown(dash_q)
        try:
            if p_dash.is_alive():
                p_dash.terminate()
        except Exception:
            pass

    print(Fore.RED + "Test finished sucessfully, storing data now..." + Style.RESET_ALL)
    print(Fore.GREEN + "Data stored sucessfully" + Style.RESET_ALL)
    print(Fore.RED + "Data being processed..." + Style.RESET_ALL)

    def remove_outliers(df, method):
        n_pre = df.shape[0]
        if method == "z-score":
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == "quantile":
            q1 = df.quantile(q=0.25)
            q3 = df.quantile(q=0.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f"{diff} rows removed {round(diff / n_pre * 100, 2)}%")
        return df

    for df_name in os.listdir(f"{folder}/Prepro/"):
        if df_name[-4:] == ".csv" and df_name[:4] != "file":
            df_base = df_name[:-4]
            df_raw = pd.read_csv(f"{folder}/Prepro/{df_base}.csv").drop(["board_ts", "unix_ts"], axis=1)
            df_processed = remove_outliers(
                df_raw.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True),
                "quantile",
            )
            df_processed.to_csv(f"{folder}/Processed/{df_base}_processed.csv")
            df_processed.plot()
            plt.savefig(f"{folder}/Figures/{df_base}_plot.png")
            plt.close()

    for df_name2 in os.listdir(f"{folder}/Prepro 2/"):
        if df_name2[-4:] == ".csv" and df_name2[:4] != "file":
            df_base2 = df_name2[:-4]
            df_raw2 = pd.read_csv(f"{folder}/Prepro 2/{df_base2}.csv").drop(["board_ts", "unix_ts"], axis=1)
            df_processed2 = remove_outliers(
                df_raw2.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True),
                "quantile",
            )
            df_processed2.to_csv(f"{folder}/Processed 2/{df_base2}_processed2.csv")
            df_processed2.plot()
            plt.savefig(f"{folder}/Figures 2/{df_base2}_plot2.png")
            plt.close()

    print(Fore.GREEN + "Data processed successfully" + Style.RESET_ALL)

    try:
        data_meanb = pd.read_csv(f"{folder}/Frequency_bands_bispectrum.csv", index_col=0)
        data_graph = data_meanb.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)

        meta_df = pd.DataFrame(
            {
                "Muse_1": [muse_code1],
                "Muse_2": [muse_code2],
                "Relación": [relationship],
                "Fecha": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Modo_final": ["csv" if source_mode.value == CSV_MODE else "live"],
            }
        )

        final_df = pd.concat([meta_df, data_graph], axis=1)
        final_df.to_csv(f"{folder}/Frequency_bands_bispectrum_with_metadata.csv", index=False)
        print(Fore.GREEN + "Archivo combinado con metadatos guardado correctamente." + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"Error al combinar metadatos con datos: {e}" + Style.RESET_ALL)
