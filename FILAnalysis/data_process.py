import os
import pandas as pd
import numpy as np
import scipy.signal as signal


# === Ruta base: carpeta donde está este script ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están TODAS las carpetas de sujetos 
RAW_ROOT = os.path.join(BASE_DIR, "Rawdata")

# Nombre de archivos dentro de cada carpeta de sujeto
RAW1_NAME = "Raw.csv"    # primera persona / derivación
RAW2_NAME = "Raw2.csv"   # segunda persona / derivación 
MARKERS_NAME = "markers.csv"

# Canales EEG a filtrar 
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Frecuencia de muestreo (Hz)
FS = 256

# Band-pass 4–50 Hz, Butterworth orden 2
SOS = signal.butter(2, [4, 50], btype="bandpass", fs=FS, output="sos")



def encontrar_sesiones(raw_root):
    """
    Recorre RAW_ROOT y regresa una lista de carpetas que tienen
    Raw.csv (o el nombre que definas) y markers.csv.
    """
    sesiones = []
    for root, dirs, files in os.walk(raw_root):
        files_set = set(files)
        if MARKERS_NAME in files_set and RAW1_NAME in files_set:
            sesiones.append(root)
    return sesiones


def cargar_csv_si_existe(folder, filename):
    ruta = os.path.join(folder, filename)
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    else:
        return None


def filtrar_df(df, eeg_cols, sos):
    """
    Aplica filtro band-pass 4–50 Hz a los canales de EEG.
    Convierte las columnas a numérico y limpia textos/NaN.
    """
    df_f = df.copy()
    cols_presentes = [c for c in eeg_cols if c in df.columns]

    if not cols_presentes:
        print("⚠ No se encontraron columnas EEG esperadas en este archivo.")
        return df_f

    # 1) Convertir a numérico forzado (texto raro -> NaN)
    datos_num = df[cols_presentes].apply(pd.to_numeric, errors="coerce")

    # 2) Rellenar NaN para que el filtro no truene
    #    Primero rellenamos hacia adelante y hacia atrás y al final 0 si sigue faltando
    datos_num = datos_num.ffill().bfill().fillna(0)


    # 3) Aplicar el filtro
    datos_filtrados = signal.sosfiltfilt(sos, datos_num.values, axis=0)

    # 4) Meter de regreso al dataframe copia
    df_f[cols_presentes] = datos_filtrados

    return df_f


def obtener_col_tiempo(df):
    """
    Intenta adivinar el nombre de la columna de tiempo.
    Ajusta aquí si sabes exactamente cómo se llama.
    """
    posibles = ["unix_ts", "timestamp", "time", "Time", "TimeStamp"]
    for col in posibles:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró columna de tiempo. Renombra aquí según tu CSV. Columnas: {df.columns.tolist()}"
    )


def segmentar_por_markers(df, markers):
    """
    Segmenta en 3 fases usando markers.

    Caso A: si existen labels 'baseline_start', 'shared_start', 'individual_start', los usa.
    Caso B: si NO existen, usa los primeros 3 markers en orden de tiempo.

    Segmentos:
      baseline:  [t_baseline, t_shared)
      shared:    [t_shared,  t_ind)
      individual:[t_ind,     fin)
    """

    # Trabajar con copias para no modificar los originales fuera
    df = df.copy()
    markers = markers.copy()

    # 1) Columna de tiempo en la señal
    t_col = obtener_col_tiempo(df)

    # 2) Columna de tiempo en markers
    m_t_col = None
    for col in ["unix_ts", "timestamp", "time", "Time", "TimeStamp"]:
        if col in markers.columns:
            m_t_col = col
            break
    if m_t_col is None:
        raise ValueError(
            f"El archivo markers no tiene columna de tiempo reconocible. Columnas: {markers.columns.tolist()}"
        )

    # 3) Asegurarnos de que ambas columnas de tiempo sean NUMÉRICAS
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    markers[m_t_col] = pd.to_numeric(markers[m_t_col], errors="coerce")

    # Rellenamos por si sale algún NaN raro
    df[t_col] = df[t_col].ffill().bfill()
    markers[m_t_col] = markers[m_t_col].ffill().bfill()

    # 4) Ordenar markers por tiempo
    m_sorted = markers.sort_values(m_t_col).reset_index(drop=True)

    # 5) Intentar usar labels estándar si existen
    def get_time_for_label(label):
        if "label" not in markers.columns:
            return None
        rows = markers[markers["label"] == label]
        if rows.empty:
            return None
        return rows[m_t_col].iloc[0]

    t_baseline = get_time_for_label("baseline_start")
    t_shared   = get_time_for_label("shared_start")
    t_ind      = get_time_for_label("individual_start")

    # 6) Fallback: si no encontramos esos labels, usamos los primeros 3 tiempos
    if t_baseline is None or t_shared is None or t_ind is None:
        if len(m_sorted) < 3:
            print("⚠ Menos de 3 marcadores; no se puede segmentar bien esta sesión. Se devuelve sólo la señal completa.")
            return {"completa": df}

        print("⚠ No se encontraron labels estándar; usando los primeros 3 markers por tiempo como baseline/shared/individual.")
        t_baseline, t_shared, t_ind = m_sorted[m_t_col].iloc[0:3]

    segs = {}

    # 7) Construir máscaras usando la columna de tiempo numérica
    ts = df[t_col]

    # baseline: [t_baseline, t_shared)
    mask_base = (ts >= t_baseline) & (ts < t_shared)
    segs["baseline"] = df.loc[mask_base].reset_index(drop=True)

    # shared: [t_shared, t_ind)
    mask_shared = (ts >= t_shared) & (ts < t_ind)
    segs["shared"] = df.loc[mask_shared].reset_index(drop=True)

    # individual: [t_ind, fin)
    mask_ind = ts >= t_ind
    segs["individual"] = df.loc[mask_ind].reset_index(drop=True)

    return segs



def procesar_sesion(session_folder):
    """
    Procesa una sola carpeta de sesión:
    - filtra Raw y Raw2 (si existe)
    - segmenta en 3 fases
    - guarda todo en subcarpeta 'processed'
    """
    print(f"\n=== Procesando sesión: {session_folder} ===")

    raw1 = cargar_csv_si_existe(session_folder, RAW1_NAME)
    raw2 = cargar_csv_si_existe(session_folder, RAW2_NAME)
    markers = cargar_csv_si_existe(session_folder, MARKERS_NAME)

    if raw1 is None or markers is None:
        print(" Falta Raw.csv o markers.csv, se salta esta sesión.")
        return

    # Carpeta de salida
    out_dir = os.path.join(session_folder, "processed")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Procesar Raw1 ----------
    print("  - Filtrando Raw (p1)...")
    raw1_f = filtrar_df(raw1, EEG_CHANNELS, SOS)

    # Guardar señal completa filtrada
    out_raw1_full = os.path.join(out_dir, "p1_filtrado_completo.csv")
    raw1_f.to_csv(out_raw1_full, index=False)
    print(f"    ✅ Guardado: {out_raw1_full}")

    # Segmentar Raw1
    print("  - Segmentando Raw (p1)...")
    segs1 = segmentar_por_markers(raw1_f, markers)
    for fase, df_seg in segs1.items():
        ruta = os.path.join(out_dir, f"p1_{fase}.csv")
        df_seg.to_csv(ruta, index=False)
        print(f"    ✅ Guardado: {ruta}")

    # ---------- Procesar Raw2 si existe ----------
    if raw2 is not None:
        print("  - Filtrando Raw2 (p2)...")
        raw2_f = filtrar_df(raw2, EEG_CHANNELS, SOS)

        out_raw2_full = os.path.join(out_dir, "p2_filtrado_completo.csv")
        raw2_f.to_csv(out_raw2_full, index=False)
        print(f"    ✅ Guardado: {out_raw2_full}")

        print("  - Segmentando Raw2 (p2)...")
        segs2 = segmentar_por_markers(raw2_f, markers)
        for fase, df_seg in segs2.items():
            ruta = os.path.join(out_dir, f"p2_{fase}.csv")
            df_seg.to_csv(ruta, index=False)
            print(f"    ✅ Guardado: {ruta}")
    else:
        print("  ⚠ No se encontró Raw2.csv, sólo se procesó Raw.csv")


# ================= MAIN =================

def main():
    print(f"Buscando sesiones en: {RAW_ROOT}")
    sesiones = encontrar_sesiones(RAW_ROOT)
    if not sesiones:
        print("No se encontraron carpetas con Raw.csv y markers.csv. Revisa RAW_ROOT.")
        return

    print(f"Se encontraron {len(sesiones)} sesión(es).")
    for folder in sesiones:
        procesar_sesion(folder)

    print("\n Listo. Todas las señales filtradas y segmentadas han sido guardadas.")


if __name__ == "__main__":
    main()
