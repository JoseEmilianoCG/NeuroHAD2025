import os
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIGURA AQUÍ LO QUE QUIERES VER ===============

# Sujeto (carpeta dentro de Rawdata)
SESSION = "S48R2"      # Ejemplos: "S1R1", "S2R2", "S5R1", "S10R1", etc.

# Persona / derivación
PERSON = "p2"          # "p1" o "p2"

# Fase:
#   "baseline", "shared", "individual"
#   o "filtrado_completo" para ver toda la señal filtrada
PHASE = "individual"

# Canal EEG a graficar
CHANNEL = "AF8"       # "TP9", "AF7", "AF8", "TP10"

# Ventana de muestras a mostrar (opcional)
#   Dejar en None para ver toda la señal
WINDOW = None          # Ejemplo: 5000 para ver solo las primeras 5000 muestras se puede cambiar si quieren ver algo más 


#Datos y configuración de la gráfica y excel (no mover a menos q se requiera)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(BASE_DIR, "Rawdata")


def construir_ruta_csv(session, person, phase):
    """
    Arma la ruta del archivo CSV ya procesado.
    """
    session_folder = os.path.join(RAW_ROOT, session, "processed")

    if phase == "filtrado_completo":
        filename = f"{person}_filtrado_completo.csv"
    else:
        filename = f"{person}_{phase}.csv"

    full_path = os.path.join(session_folder, filename)
    return full_path


def main():
    print("=== VISUALIZADOR DE SEÑALES FILTRADAS ===")
    print(f"Sujeto: {SESSION} | Persona: {PERSON} | Fase: {PHASE} | Canal: {CHANNEL}")

    csv_path = construir_ruta_csv(SESSION, PERSON, PHASE)

    if not os.path.exists(csv_path):
        print(f" No se encontró el archivo:\n  {csv_path}")
        print("Revisa que la sesión, persona y fase sean correctas.")
        return

    print(f"📂 Cargando: {csv_path}")
    df = pd.read_csv(csv_path)

    if CHANNEL not in df.columns:
        print(f" El canal '{CHANNEL}' no está en las columnas del CSV.")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        return

    y = df[CHANNEL].values

    if WINDOW is not None:
        y = y[:WINDOW]

    plt.figure(figsize=(12, 4))
    plt.plot(y)
    plt.title(f"{SESSION} - {PERSON} - {PHASE} - {CHANNEL}")
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud (uV aprox)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
