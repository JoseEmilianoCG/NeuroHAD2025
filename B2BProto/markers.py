import csv
import os
import time
import sys

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
    import os
    import select
    import termios
    import tty

    class _TTY:
        """
        Abre /dev/tty (independiente de sys.stdin) y pone el terminal en cbreak
        durante su vida útil, restaurándolo al salir.
        """
        def __init__(self):
            # Si no hay TTY, lanzará OSError
            self.fd = os.open("/dev/tty", os.O_RDONLY)
            self.old = termios.tcgetattr(self.fd)

        def __enter__(self):
            # cbreak: sin canonico y sin eco, pero conservando señales
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, exc_type, exc, tb):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            os.close(self.fd)

        def kbhit(self):
            dr, _, _ = select.select([self.fd], [], [], 0)
            return bool(dr)

        def getch(self):
            # lee 1 byte sin bloqueo (ya estamos en cbreak)
            return os.read(self.fd, 1).decode(errors="ignore")

    # Instancia global para usar en el bucle
    _tty_ctx = None

    def kbhit():
        global _tty_ctx
        if _tty_ctx is None:
            raise RuntimeError("Inicializa el contexto TTY primero (ver marker_loop).")
        return _tty_ctx.kbhit()

    def getch():
        global _tty_ctx
        if _tty_ctx is None:
            raise RuntimeError("Inicializa el contexto TTY primero (ver marker_loop).")
        return _tty_ctx.getch()


# --- main function ---
def marker_loop(csv_path, label_map=None):
    """
    Escucha teclas y escribe marcas: [unix_ts, key, label]
    - En macOS/Linux usa /dev/tty en modo cbreak (no requiere Enter).
    - En Windows usa msvcrt.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    is_new = not os.path.exists(csv_path)

    # Prepara CSV
    f = open(csv_path, 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    if is_new:
        w.writerow(['unix_ts', 'key', 'label'])
        f.flush()

    try:
        if _IS_WINDOWS:
            # Windows: no hace falta contexto
            while True:
                if kbhit():
                    key = getch()
                    ts = time.time()
                    label = (label_map or {}).get(key, key)
                    w.writerow([ts, key, label])
                    f.flush()
                time.sleep(0.01)
        else:
            # Unix/macOS: abre TTY y entra a cbreak para que kbhit funcione
            global _tty_ctx
            with _TTY() as ttyctx:
                _tty_ctx = ttyctx
                while True:
                    if kbhit():
                        key = getch()
                        ts = time.time()
                        label = (label_map or {}).get(key, key)
                        w.writerow([ts, key, label])
                        f.flush()
                    time.sleep(0.01)
    finally:
        try:
            f.close()
        except Exception:
            pass
