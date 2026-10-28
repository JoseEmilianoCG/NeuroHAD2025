import csv
import os
import time
import sys
import threading
import queue
from datetime import datetime
import tkinter as tk
from tkinter import ttk

# --- cross-platform kbhit/getch ---
try:
    import msvcrt

    _IS_WINDOWS = True
except ImportError:
    _IS_WINDOWS = False

if _IS_WINDOWS:

    def kbhit():
        return msvcrt.kbhit()

    def getch():
        # Unicode-safe
        return msvcrt.getwch()

else:
    import select
    import termios
    import tty

    class _TTY:
        """
        Abre /dev/tty y pone el terminal en cbreak durante su vida útil,
        restaurándolo al salir.
        """

        def __init__(self):
            # Si no hay TTY, lanzará OSError
            self.fd = os.open("/dev/tty", os.O_RDONLY)
            self.old = termios.tcgetattr(self.fd)

        def __enter__(self):
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, exc_type, exc, tb):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            os.close(self.fd)

        def kbhit(self):
            dr, _, _ = select.select([self.fd], [], [], 0)
            return bool(dr)

        def getch(self):
            return os.read(self.fd, 1).decode(errors="ignore")

    # Por comodidad, instanciamos dentro del hilo lector.


def _ensure_parent(path: str):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def _append_txt_line(txt_path: str, label: str):
    """Escribe una línea del tipo: "LABEL" da inicio"""
    if not txt_path:
        return
    _ensure_parent(txt_path)
    with open(txt_path, "a", encoding="utf-8") as tf:
        tf.write(f'"{label}" da inicio\n')


def _append_csv_row(csv_path: str, key: str, label: str):
    """Escribe [unix_ts, key, label] en CSV (crea encabezado si no existe)."""
    _ensure_parent(csv_path)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["unix_ts", "key", "label"])
        w.writerow([time.time(), key, label])


def marker_listener(
    csv_path, label_map, txt_path, msg_queue: queue.Queue, stop_event: threading.Event
):
    """
    Hilo lector de teclado. Empuja mensajes a la cola para la UI y
    escribe en CSV (+ opcional TXT).
    """
    try:
        if _IS_WINDOWS:
            while not stop_event.is_set():
                if kbhit():
                    key = getch()
                    label = (label_map or {}).get(key, key)
                    _append_csv_row(csv_path, key, label)
                    _append_txt_line(txt_path, label)
                    msg_queue.put(f'"{label}" da inicio')
                time.sleep(0.01)
        else:
            with _TTY() as ttyctx:
                while not stop_event.is_set():
                    if ttyctx.kbhit():
                        key = ttyctx.getch()
                        label = (label_map or {}).get(key, key)
                        _append_csv_row(csv_path, key, label)
                        _append_txt_line(txt_path, label)
                        msg_queue.put(f'"{label}" da inicio')
                    time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        msg_queue.put(f"[ERROR] {e!r}")
    finally:
        msg_queue.put("[INFO] Lector detenido.")


# ----------------- UI (Tkinter) -----------------


class LiveLogWindow:
    def __init__(self, title="Histórico de estados", width=640, height=380):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(420, 280)

        # Encabezado
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.status_lbl = ttk.Label(
            top, text="Presiona teclas para registrar…", font=("Segoe UI", 10)
        )
        self.status_lbl.pack(side="left")

        # Área de texto con scrollbar
        mid = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        mid.pack(fill="both", expand=True)

        self.text = tk.Text(mid, wrap="word", state="disabled", font=("Consolas", 11))
        self.scroll = ttk.Scrollbar(mid, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)

        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        # Barra inferior
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        self.time_lbl = ttk.Label(bottom, text="")
        self.time_lbl.pack(side="right")

        # Cola para mensajes entrantes
        self.msg_queue = queue.Queue()
        self._poll_interval_ms = 50
        self._alive = True

        # Reloj
        self._tick()

        # Cierre
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _tick(self):
        if not self._alive:
            return
        self.time_lbl.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self._tick)

    def append_line(self, line: str):
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def pump_queue(self):
        """Vacía la cola de mensajes hacia la UI."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.append_line(msg)
        except queue.Empty:
            pass
        if self._alive:
            self.root.after(self._poll_interval_ms, self.pump_queue)

    def _on_close(self):
        self._alive = False
        self.root.quit()

    def run(self):
        self.pump_queue()
        self.root.mainloop()


def run_with_window(csv_path, label_map=None, txt_path=None):
    """
    Lanza la ventana (en el hilo principal) y el lector de teclado en un hilo aparte.
    """
    # Construye la UI
    ui = LiveLogWindow(title="Histórico de estados")

    # Sincronización con el hilo lector
    stop_event = threading.Event()

    # Hilo lector
    t = threading.Thread(
        target=marker_listener,
        args=(csv_path, label_map, txt_path, ui.msg_queue, stop_event),
        daemon=True,
    )
    t.start()

    # Mensajes iniciales
    ui.append_line("[INFO] Ventana lista. Comienza a presionar teclas…")
    if txt_path:
        ui.append_line(f"[INFO] Guardando histórico en: {txt_path}")
    ui.append_line(f"[INFO] Guardando CSV en: {csv_path}")

    try:
        ui.run()
    finally:
        stop_event.set()
        t.join(timeout=1.0)


# ----------------- Ejecución directa -----------------
if __name__ == "__main__":
    labels = {
        "1": "Reposo",
        "2": "Actividad 1",
        "3": "Actividad 2",
        "4": "Actividad 3",
        "r": "Reposo",
        "a": "Actividad 1",
        "s": "Actividad 2",
        "d": "Actividad 3",
    }
    csv_out = "logs/markers.csv"
    txt_out = "logs/historico.txt"  # opcional; pon None para no escribir TXT
    try:
        run_with_window(csv_out, label_map=labels, txt_path=txt_out)
    except KeyboardInterrupt:
        print("\nSaliendo por Ctrl+C")
