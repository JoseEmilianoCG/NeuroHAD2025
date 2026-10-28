import threading
from connection_popup import LiveConnectionWindow


def run_connection_popup(mp_queue, title="Estado de conexión Muse"):
    win = LiveConnectionWindow(title=title)
    win.append_line("[INFO] Ventana de conexión lista.")

    def forward_messages():
        while True:
            msg = mp_queue.get()
            if msg is None:
                break
            win.append_line(str(msg))

    threading.Thread(target=forward_messages, daemon=True).start()
    win.run()
