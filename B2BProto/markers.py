import msvcrt
import csv
import os 
import time

def marker_loop(csv_path, label_map=None):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['unix_ts', 'key', 'label'])
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()           # lee una tecla
                ts  = time.time()
                label = (label_map or {}).get(key, key)
                w.writerow([ts, key, label])
                f.flush()
            time.sleep(0.01)