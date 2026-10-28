import time

STATUS_STOPPED = 0
STATUS_CONNECTED = 1
STATUS_RECONNECTING = 2
STATUS_DISCONNECTED = 3
STATUS_CSV = 4
STATUS_ERROR = 5


def _emit(ui_queue, text):
    if ui_queue is None:
        return
    try:
        ui_queue.put(text)
    except Exception:
        pass


def connection_monitor(
    second,
    totaltime,
    source_mode,
    heartbeat1,
    heartbeat2,
    reconnect_req1,
    reconnect_req2,
    status1,
    status2,
    names=None,
    stale_timeout=8.0,
    poll_sec=1.0,
    ui_queue=None,
):
    if names is None:
        names = ("Muse-1", "Muse-2")

    last_alert = {0: 0.0, 1: 0.0}

    while True:
        with second.get_lock():
            if second.value >= totaltime:
                _emit(ui_queue, "[monitor] Fin del monitoreo de conexión.")
                return

        with source_mode.get_lock():
            mode = source_mode.value

        if mode != 0:
            time.sleep(poll_sec)
            continue

        now = time.time()
        devs = [
            (0, names[0], heartbeat1, reconnect_req1, status1),
            (1, names[1], heartbeat2, reconnect_req2, status2),
        ]

        for idx, name, hb, req, st in devs:
            with hb.get_lock():
                last_seen = hb.value
            with st.get_lock():
                status_code = st.value

            if status_code in (STATUS_CSV, STATUS_STOPPED):
                continue
            if last_seen <= 0:
                continue

            age = now - last_seen
            if age >= stale_timeout:
                with req.get_lock():
                    already_requested = req.value == 1
                    if not already_requested:
                        req.value = 1
                        msg = (
                            f"[monitor] {name} sin datos desde hace {age:.1f}s. "
                            "Se solicita reconexión."
                        )
                        print(msg)
                        _emit(ui_queue, msg)
                if now - last_alert[idx] >= stale_timeout:
                    last_alert[idx] = now

        time.sleep(poll_sec)
