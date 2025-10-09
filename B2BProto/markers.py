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
        ch = msvcrt.getwch()  # unicode
        return ch
else:
    import select
    import termios
    import tty

    def kbhit():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(dr)

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

# --- main function ---
def marker_loop(csv_path, label_map=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['unix_ts', 'key', 'label'])
        while True:
            if kbhit():
                key = getch()
                ts = time.time()
                label = (label_map or {}).get(key, key)
                w.writerow([ts, key, label])
                f.flush()
            time.sleep(0.01)
