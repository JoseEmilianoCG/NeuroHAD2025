from bispectrum import bispec
from timer import timer
from markers import run_with_window
from neuro_dashboard import run_dash, shutdown
from csv_switch_provider_reconnect_ui import (
    device_provider,
    LIVE_MODE,
    CSV_MODE,
    STATUS_RECONNECTING,
)
from connection_monitor_ui import connection_monitor
from connection_popup_runner import run_connection_popup

import multiprocessing
from multiprocessing import Queue, Process, Value, Manager
import threading

from colorama import Fore, Style

import os
from datetime import datetime
import pandas as pd

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


def keyboard_listener(source_mode, conn_ui_queue=None):
    print("\nControles de captura:")
    print("  c + Enter  -> cambiar una sola vez de LIVE a CSV")
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            return
        if cmd == "c":
            with source_mode.get_lock():
                if source_mode.value == LIVE_MODE:
                    source_mode.value = CSV_MODE
                    msg = "Cambio a modo CSV solicitado."
                    print(Fore.YELLOW + msg + Style.RESET_ALL)
                    if conn_ui_queue is not None:
                        conn_ui_queue.put(f"[usuario] {msg}")
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
    conn_ui_queue = Queue()
    source_mode = Value("i", LIVE_MODE)

    heartbeat1 = Value("d", 0.0)
    heartbeat2 = Value("d", 0.0)
    reconnect_req1 = Value("i", 0)
    reconnect_req2 = Value("i", 0)
    status1 = Value("i", STATUS_RECONNECTING)
    status2 = Value("i", STATUS_RECONNECTING)

    session_num, repetition_num = input(
        "Introduce el número de sesión y repetición, separa con un espacio: "
    ).split(" ")
    session_num = f"{int(session_num):02d}"
    repetition_num = f"{int(repetition_num):02d}"
    basaltime = int(input("Introduce el tiempo de baseline en segundos: "))
    totaltime = int(input("Introduce el tiempo total de la sesión en segundos: "))

    print("\n=== Información adicional del experimento ===")
    muse_code1 = "Muse-" + str(input("ID de Muse #1: ")).upper()
    muse_code2 = "Muse-" + str(input("ID de Muse #2: ")).upper()
    relationship = input("Relación entre los sujetos: ")

    csv_path1 = r'C:\Users\josee\OneDrive\Documentos\GitHub\NeuroHAD2025\B2BProto_V2\simul\Raw.csv'
    csv_path2 = r'C:\Users\josee\OneDrive\Documentos\GitHub\NeuroHAD2025\B2BProto_V2\simul\Raw2.csv'

    metadata_df = pd.DataFrame([
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
    ])

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
    timestamps = mgr.list()

    threading.Thread(
        target=keyboard_listener,
        args=(source_mode, conn_ui_queue),
        daemon=True,
    ).start()

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

    q_conn_ui = Process(
        target=run_connection_popup,
        args=(conn_ui_queue, "Estado de conexión Muse"),
        daemon=True,
    )

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
            heartbeat1,
            status1,
            reconnect_req1,
        ],
        kwargs=dict(
            prepro_subfolder="Prepro",
            raw_filename="Raw.csv",
            clean_filename="Clean.csv",
            chunk_sec=sleeptime,
            ui_queue=conn_ui_queue,
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
            heartbeat2,
            status2,
            reconnect_req2,
        ],
        kwargs=dict(
            prepro_subfolder="Prepro 2",
            raw_filename="Raw2.csv",
            clean_filename="Clean2.csv",
            chunk_sec=sleeptime,
            ui_queue=conn_ui_queue,
        ),
    )
    qmon = Process(
        target=connection_monitor,
        args=[
            seconds,
            totaltime,
            source_mode,
            heartbeat1,
            heartbeat2,
            reconnect_req1,
            reconnect_req2,
            status1,
            status2,
        ],
        kwargs=dict(
            names=(muse_code1, muse_code2),
            stale_timeout=8.0,
            poll_sec=1.0,
            ui_queue=conn_ui_queue,
        ),
        daemon=True,
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
        p_dash.start()
        q_conn_ui.start()
        process2.start()
        q.start()
        q2.start()
        qmon.start()
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

        try:
            conn_ui_queue.put(None)
            if q_conn_ui.is_alive():
                q_conn_ui.join(timeout=2)
        except Exception:
            pass

    print(Fore.RED + "Test finished sucessfully, storing data now..." + Style.RESET_ALL)
