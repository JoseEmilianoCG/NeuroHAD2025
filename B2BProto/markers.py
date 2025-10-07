#import msvcrt
import csv
import os 
import time
import sys, select, termios, tty

def kbhit():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def marker_loop(csv_path, label_map=None):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['unix_ts', 'key', 'label'])
        while True:
            if kbhit():   #original msvcrt.kbhit()
                key = getch()           # lee una tecla.  msvcrt.getwch()
                ts  = time.time()
                label = (label_map or {}).get(key, key)
                w.writerow([ts, key, label])
                f.flush()
            time.sleep(0.01)