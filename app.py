#!/usr/bin/env python3
"""Spatial Detector - tkinter touchscreen GUI for a Raspberry Pi + Luxonis OAK-D.

Live RGB preview with on-device stereo spatial object detection (X/Y/Z per object).
Switch between detection models from a touch-friendly picker; the DepthAI pipeline
rebuilds on the fly. Photo / video / time-lapse capture saved to a session folder.

Run:
    python3 app.py                 # fullscreen, uses config.json
    python3 app.py --windowed      # windowed (testing over HDMI/VNC)
    python3 app.py --mock          # synthetic frames, no OAK required

Touch bar: MODEL | PHOTO | REC/STOP | TIME-LAPSE | EXIT.   (Esc / q also quit.)
"""

import argparse
import os
import queue
import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk

from camera import CameraWorker
from config import load_config
from models import discover_models, pick_default

HERE = os.path.dirname(os.path.abspath(__file__))
BAR_H = 90  # touch button-bar height (px)


class App:
    def __init__(self, root, cfg, mock=False):
        self.root = root
        self.cfg = cfg
        self.W, self.H = cfg["screen_size"]
        self.video_h = self.H - BAR_H

        self.entries = discover_models(cfg.get("zoo_models", []))
        current = pick_default(self.entries, cfg.get("default_model", ""))

        # The worker renders frames sized to the video area (above the bar).
        worker_cfg = dict(cfg)
        worker_cfg["screen_size"] = [self.W, self.video_h]
        self.frame_q = queue.Queue(maxsize=1)
        self.worker = CameraWorker(worker_cfg, current, self.frame_q, mock=mock)

        self._photo_ref = None     # keep ImageTk ref alive
        self._last_btn_state = None
        self._build_ui()

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("q", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.worker.start()
        self._render()

    # --- UI construction ------------------------------------------------ #

    def _build_ui(self):
        self.root.configure(bg="black")
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.video_h,
                                bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw")

        bar = tk.Frame(self.root, bg="black", height=BAR_H)
        bar.pack(side="bottom", fill="x")
        self.btn_font = tkfont.Font(family="DejaVu Sans", size=14, weight="bold")

        self.buttons = {}
        specs = [("model", "MODEL", "#3a3a3a", self.open_picker),
                 ("photo", "PHOTO", "#3a3a3a", lambda: self.worker.send("photo")),
                 ("record", "REC", "#3a3a3a", lambda: self.worker.send("record")),
                 ("timelapse", "TIME-LAPSE", "#3a3a3a",
                  lambda: self.worker.send("timelapse")),
                 ("exit", "EXIT", "#3a3a3a", self.quit)]
        for i, (key, label, color, cmd) in enumerate(specs):
            b = tk.Button(bar, text=label, command=cmd, font=self.btn_font,
                          fg="white", bg=color, activebackground="#555",
                          activeforeground="white", bd=0, relief="flat",
                          highlightthickness=0)
            b.place(relx=i / len(specs), rely=0, relwidth=1 / len(specs),
                    relheight=1.0)
            self.buttons[key] = b

    # --- model picker modal --------------------------------------------- #

    def open_picker(self):
        if not self.entries:
            return
        top = tk.Toplevel(self.root, bg="black")
        top.overrideredirect(True)
        top.geometry(f"{self.W}x{self.H}+0+0")
        top.transient(self.root)
        top.grab_set()

        big = tkfont.Font(family="DejaVu Sans", size=16, weight="bold")
        tk.Label(top, text="SELECT MODEL", font=big, fg="white",
                 bg="black").pack(pady=(16, 8))

        list_frame = tk.Frame(top, bg="black")
        list_frame.pack(fill="both", expand=True, padx=20)
        for entry in self.entries:
            kind = "zoo" if entry.kind == "zoo" else "local"
            tk.Button(list_frame, text=f"{entry.name}   [{kind}]", font=big,
                      fg="white", bg="#2a2a2a", activebackground="#4a7",
                      bd=0, relief="flat", anchor="w", padx=20,
                      command=lambda e=entry, t=top: self._choose(e, t)
                      ).pack(fill="x", pady=4, ipady=14)

        tk.Button(top, text="CANCEL", font=big, fg="white", bg="#622",
                  activebackground="#933", bd=0, relief="flat",
                  command=top.destroy).pack(fill="x", padx=20, pady=16, ipady=12)

    def _choose(self, entry, top):
        top.destroy()
        self.worker.send("switch", entry)

    # --- render loop ---------------------------------------------------- #

    def _render(self):
        try:
            frame = self.frame_q.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfig(self.canvas_img, image=img)
            self._photo_ref = img  # prevent GC

        self._refresh_buttons()
        self.root.after(15, self._render)

    def _refresh_buttons(self):
        st = self.worker.get_status()
        state = (st["recording"], st["timelapse"], st["loading"])
        if state == self._last_btn_state:
            return
        self._last_btn_state = state

        rec = self.buttons["record"]
        rec.config(text="STOP" if st["recording"] else "REC",
                   bg="#a33" if st["recording"] else "#3a3a3a")
        self.buttons["timelapse"].config(
            bg="#4a7" if st["timelapse"] else "#3a3a3a")
        self.buttons["model"].config(
            text="LOADING..." if st["loading"] else "MODEL",
            bg="#777" if st["loading"] else "#3a3a3a")

    # --- shutdown ------------------------------------------------------- #

    def quit(self):
        try:
            self.worker.stop()
        finally:
            self.root.after(150, self.root.destroy)


def main():
    ap = argparse.ArgumentParser(description="Spatial detection touchscreen app")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--windowed", action="store_true",
                    help="windowed mode (for HDMI/VNC testing)")
    ap.add_argument("--mock", action="store_true",
                    help="synthetic frames, no OAK required")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.windowed:
        cfg["fullscreen"] = False

    root = tk.Tk()
    root.title("Spatial Detector")
    w, h = cfg["screen_size"]
    if cfg.get("fullscreen", True) and not args.windowed:
        root.attributes("-fullscreen", True)
        root.config(cursor="none")
        # Match the geometry to the panel so layout maths line up.
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        cfg["screen_size"] = [root.winfo_screenwidth(), root.winfo_screenheight()]
    else:
        root.geometry(f"{w}x{h}")

    App(root, cfg, mock=args.mock)
    root.mainloop()


if __name__ == "__main__":
    main()
