import queue
import tkinter as tk
from tkinter import ttk
from datetime import datetime


class LiveConnectionWindow:
    def __init__(self, title="Estado de conexión Muse", width=760, height=340):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(480, 280)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.status_lbl = ttk.Label(
            top,
            text="Monitoreando conexión...",
            font=("Segoe UI", 10),
        )
        self.status_lbl.pack(side="left")

        mid = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        mid.pack(fill="both", expand=True)

        self.text = tk.Text(mid, wrap="word", state="disabled", font=("Consolas", 11))
        self.scroll = ttk.Scrollbar(mid, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        self.time_lbl = ttk.Label(bottom, text="")
        self.time_lbl.pack(side="right")

        self.msg_queue = queue.Queue()
        self._alive = True
        self._poll_interval_ms = 100
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _tick(self):
        if not self._alive:
            return

        self.time_lbl.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        while True:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            self._append(msg)

        self.root.after(self._poll_interval_ms, self._tick)

    def _append(self, msg):
        now = datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {msg}\n"

        self.text.configure(state="normal")
        self.text.insert("end", line)
        self.text.see("end")
        self.text.configure(state="disabled")
        self.status_lbl.config(text=msg)

    def append_line(self, msg):
        self.msg_queue.put(msg)

    def _on_close(self):
        self._alive = False
        self.root.quit()

    def run(self):
        self.root.mainloop()
