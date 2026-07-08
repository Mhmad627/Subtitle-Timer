"""Subtitle Timer — desktop GUI.

Pick a video, drag/resize the green box onto the subtitle area (scrub to a
moment where a subtitle is visible), and click Detect. The app samples
frames, asks the detector "is a subtitle visible in the box?" per frame,
groups the positives into timed blocks, and writes an .srt with placeholder
text. Until your trained model (.onnx) is selected, a simple edge-density
heuristic stands in.

Run directly:   python gui.py
Or build an exe: pyinstaller subtitle_extractor.spec
"""

import base64
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import cv2

from detector.subtitle_region import DEFAULT_CROP, clamp_rect
from output.srt_writer import write_srt
import main as pipeline

# When built as a windowed exe (no console), sys.stdout/sys.stderr are None.
# Patch them to a null writer immediately so any import-time or early output
# doesn't crash with "'NoneType' object has no attribute 'write'".
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

PREVIEW_MAX_W = 480
PREVIEW_MAX_H = 270
BOX_COLOR = "#00e000"
HANDLE_SIZE = 8        # px, corner squares
GRAB_RADIUS = 10       # px, how close to a corner counts as grabbing it
MIN_BOX_PX = 12        # px, minimum box width/height while resizing


class _QueueWriter:
    """File-like object that forwards writes to a queue (thread -> GUI).

    If constructed with a cancel event, write() raises PipelineCancelled once
    the event is set, aborting long operations at their next output.
    """

    def __init__(self, q, cancel_event=None):
        self._q = q
        self._cancel_event = cancel_event
        self.armed = cancel_event is not None

    def write(self, text):
        if self.armed and self._cancel_event.is_set():
            raise pipeline.PipelineCancelled()
        if text:
            self._q.put(text)

    def flush(self):
        pass

    def isatty(self):
        return False


class _Tooltip:
    """Small tooltip shown when hovering (or clicking) a widget."""

    def __init__(self, widget, text, wraplength=340):
        self._widget = widget
        self._text = text
        self._wraplength = wraplength
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Button-1>", self._show)

    def _show(self, _event=None):
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip,
            text=self._text,
            justify="left",
            wraplength=self._wraplength,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
        ).pack()

    def _hide(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


_HELP = {
    "fps": (
        "How many frames per second are checked for a subtitle. Higher gives "
        "tighter start/end timing but is proportionally slower. Detection is "
        "cheap, so 5 is a good default (0.2 s timing precision)."
    ),
    "threshold": (
        "Detector score cutoff, 0-1. A frame counts as 'subtitle visible' "
        "when the detector's score is at least this. Lower catches fainter "
        "subs but risks false positives; higher is stricter."
    ),
    "text": (
        "Placeholder text written into every .srt block. This tool times "
        "subtitles, it doesn't read them — fill the text in later with a "
        "subtitle editor (e.g. Aegisub)."
    ),
    "model": (
        "Your trained subtitle-presence model (.onnx, see TRAINING.md). "
        "Leave blank to use the built-in edge-density heuristic — a crude "
        "stand-in that works best on high-contrast subs."
    ),
}


class SubtitleTimerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Subtitle Timer")
        root.geometry("720x800")
        root.minsize(640, 680)

        self._log_queue = queue.Queue()
        self._worker = None
        self._cancel_event = None

        self._preview_frame_bgr = None    # cached full-res BGR frame
        self._preview_photo = None        # keep a ref or Tk drops the image
        self._preview_job = None          # pending debounced reload
        self._img_size = None             # displayed image (w, h) in px
        self._crop = list(DEFAULT_CROP)   # (x, y, w, h) as frame fractions
        self._drag = None                 # active drag state, see _on_box_press

        self._build_widgets()
        self._poll_log_queue()

    # ---------- layout ----------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        # Input file row
        file_frame = ttk.LabelFrame(self.root, text="Video file")
        file_frame.pack(fill="x", **pad)

        self.input_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_var).pack(
            side="left", fill="x", expand=True, padx=6, pady=6
        )
        ttk.Button(file_frame, text="Browse...", command=self._browse_input).pack(
            side="left", padx=6, pady=6
        )

        # Output file row
        out_frame = ttk.LabelFrame(self.root, text="Output .srt (blank = next to video)")
        out_frame.pack(fill="x", **pad)

        self.output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=6, pady=6
        )
        ttk.Button(out_frame, text="Save as...", command=self._browse_output).pack(
            side="left", padx=6, pady=6
        )

        # Model row
        model_frame = ttk.LabelFrame(
            self.root, text="Detection model (.onnx) — blank = built-in heuristic"
        )
        model_frame.pack(fill="x", **pad)

        self.model_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_var).pack(
            side="left", fill="x", expand=True, padx=6, pady=6
        )
        ttk.Button(model_frame, text="Browse...", command=self._browse_model).pack(
            side="left", padx=6, pady=6
        )
        info = ttk.Label(model_frame, text="ⓘ", foreground="#1a6fd4", cursor="hand2")
        info.pack(side="left", padx=(0, 8))
        _Tooltip(info, _HELP["model"])

        # Preview with the draggable subtitle box
        prev_frame = ttk.LabelFrame(
            self.root, text="Preview — drag the green box onto the subtitle area"
        )
        prev_frame.pack(fill="x", **pad)

        self.canvas = tk.Canvas(
            prev_frame, width=PREVIEW_MAX_W, height=140,
            background="#1e1e1e", highlightthickness=0,
        )
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._on_box_press)
        self.canvas.bind("<B1-Motion>", self._on_box_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_box_release)
        self.canvas.bind("<Motion>", self._on_box_hover)
        self._draw_placeholder("No video selected.")

        scrub_row = ttk.Frame(prev_frame)
        scrub_row.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(scrub_row, text="Scrub:").pack(side="left")
        self.scrub_var = tk.DoubleVar(value=50.0)
        ttk.Scale(
            scrub_row, from_=0.0, to=100.0, variable=self.scrub_var,
            command=lambda _v: self._schedule_preview_load(),
        ).pack(side="left", fill="x", expand=True, padx=6)

        self.box_var = tk.StringVar()
        ttk.Label(prev_frame, textvariable=self.box_var).pack(
            anchor="w", padx=6, pady=(0, 6)
        )
        self._update_box_readout()

        # Options
        opt = ttk.LabelFrame(self.root, text="Options")
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="Sample fps:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        self.fps_var = tk.StringVar(value="5")
        ttk.Entry(opt, textvariable=self.fps_var, width=8).grid(
            row=0, column=1, sticky="w", padx=6, pady=4
        )
        self._add_info(opt, 0, 2, _HELP["fps"])

        ttk.Label(opt, text="Threshold:").grid(row=0, column=3, sticky="e", padx=6, pady=4)
        self.threshold_var = tk.StringVar(value="0.5")
        ttk.Entry(opt, textvariable=self.threshold_var, width=8).grid(
            row=0, column=4, sticky="w", padx=6, pady=4
        )
        self._add_info(opt, 0, 5, _HELP["threshold"])

        ttk.Label(opt, text="Block text:").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self.text_var = tk.StringVar(value="...")
        ttk.Entry(opt, textvariable=self.text_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=4
        )
        self._add_info(opt, 1, 2, _HELP["text"])

        # Run/Cancel buttons + status
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_frame, text="Detect Subtitles", command=self._on_run)
        self.run_btn.pack(side="left", padx=6)
        self.cancel_btn = ttk.Button(
            run_frame, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=10)

        # Log output
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, wrap="word", height=10, state="disabled")
        self.log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side="right", fill="y", pady=6)
        self.log.config(yscrollcommand=scroll.set)

        self.input_var.trace_add("write", lambda *_: self._schedule_preview_load())

    def _add_info(self, parent, row, column, text):
        label = ttk.Label(parent, text="ⓘ", foreground="#1a6fd4", cursor="hand2")
        label.grid(row=row, column=column, sticky="w", padx=(0, 10))
        _Tooltip(label, text)

    # ---------- preview ----------

    def _schedule_preview_load(self):
        """Debounce reloads (typing a path / dragging the scrub bar)."""
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(150, self._load_preview_frame)

    def _load_preview_frame(self):
        self._preview_job = None
        path = self.input_var.get().strip()
        if not os.path.isfile(path):
            self._draw_placeholder("No video selected.")
            return

        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                self._draw_placeholder("Could not open this file as a video.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                index = int((self.scrub_var.get() / 100.0) * (total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
        finally:
            cap.release()

        if not ok or frame is None:
            self._draw_placeholder("Could not read a frame from this video.")
            return

        self._preview_frame_bgr = frame
        self._render_preview()

    def _draw_placeholder(self, message):
        self._preview_frame_bgr = None
        self._preview_photo = None
        self._img_size = None
        self.canvas.config(width=PREVIEW_MAX_W, height=140)
        self.canvas.delete("all")
        self.canvas.create_text(
            PREVIEW_MAX_W // 2, 70, text=message, fill="#cccccc"
        )

    def _render_preview(self):
        """Draw the cached frame on the canvas and the crop box on top."""
        frame = self._preview_frame_bgr
        if frame is None:
            return

        h, w = frame.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        img = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        ih, iw = img.shape[:2]

        ok, png = cv2.imencode(".png", img)
        if not ok:
            return
        self._preview_photo = tk.PhotoImage(
            data=base64.b64encode(png.tobytes()).decode("ascii")
        )
        self._img_size = (iw, ih)
        self.canvas.config(width=iw, height=ih)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._preview_photo, anchor="nw")
        self._draw_crop_box()

    # ---------- crop box ----------

    def _box_px(self):
        """Current crop box in canvas pixels: (x0, y0, x1, y1)."""
        iw, ih = self._img_size
        x, y, w, h = self._crop
        return (x * iw, y * ih, (x + w) * iw, (y + h) * ih)

    def _set_box_px(self, x0, y0, x1, y1):
        """Store a pixel box back as clamped frame fractions."""
        iw, ih = self._img_size
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        self._crop = list(clamp_rect(
            (x0 / iw, y0 / ih, (x1 - x0) / iw, (y1 - y0) / ih)
        ))
        self._update_box_readout()

    def _draw_crop_box(self):
        self.canvas.delete("cropbox")
        if self._img_size is None:
            return
        x0, y0, x1, y1 = self._box_px()
        self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=BOX_COLOR, width=2, tags="cropbox"
        )
        r = HANDLE_SIZE / 2
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.canvas.create_rectangle(
                cx - r, cy - r, cx + r, cy + r,
                outline=BOX_COLOR, fill=BOX_COLOR, tags="cropbox",
            )

    def _update_box_readout(self):
        x, y, w, h = self._crop
        self.box_var.set(
            f"Box: x={x:.2f}  y={y:.2f}  w={w:.2f}  h={h:.2f}   "
            "(drag inside to move, corners to resize)"
        )

    def _hit_test(self, ex, ey):
        """What is under the mouse: ('corner', anchor_px) / 'inside' / None."""
        if self._img_size is None:
            return None
        x0, y0, x1, y1 = self._box_px()
        corners = {
            (x0, y0): (x1, y1),
            (x1, y0): (x0, y1),
            (x0, y1): (x1, y0),
            (x1, y1): (x0, y0),
        }
        for (cx, cy), anchor in corners.items():
            if abs(ex - cx) <= GRAB_RADIUS and abs(ey - cy) <= GRAB_RADIUS:
                return ("corner", anchor)
        if x0 <= ex <= x1 and y0 <= ey <= y1:
            return "inside"
        return None

    def _on_box_hover(self, event):
        hit = self._hit_test(event.x, event.y)
        if hit == "inside":
            self.canvas.config(cursor="fleur")
        elif hit is not None:
            self.canvas.config(cursor="sizing")
        else:
            self.canvas.config(cursor="")

    def _on_box_press(self, event):
        hit = self._hit_test(event.x, event.y)
        if hit == "inside":
            x0, y0, _x1, _y1 = self._box_px()
            self._drag = ("move", event.x - x0, event.y - y0)
        elif hit is not None:
            _kind, anchor = hit
            self._drag = ("resize", anchor)
        else:
            self._drag = None

    def _on_box_drag(self, event):
        if self._drag is None or self._img_size is None:
            return
        iw, ih = self._img_size
        ex = min(max(event.x, 0), iw)
        ey = min(max(event.y, 0), ih)

        if self._drag[0] == "move":
            _mode, off_x, off_y = self._drag
            x0, y0, x1, y1 = self._box_px()
            bw, bh = x1 - x0, y1 - y0
            nx0 = min(max(ex - off_x, 0), iw - bw)
            ny0 = min(max(ey - off_y, 0), ih - bh)
            self._set_box_px(nx0, ny0, nx0 + bw, ny0 + bh)
        else:  # resize: box spans from the fixed opposite corner to the mouse
            _mode, (ax, ay) = self._drag
            if abs(ex - ax) < MIN_BOX_PX:
                ex = ax + MIN_BOX_PX if ex >= ax else ax - MIN_BOX_PX
            if abs(ey - ay) < MIN_BOX_PX:
                ey = ay + MIN_BOX_PX if ey >= ay else ay - MIN_BOX_PX
            self._set_box_px(ax, ay, ex, ey)

        self._draw_crop_box()

    def _on_box_release(self, _event):
        self._drag = None

    # ---------- actions ----------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save .srt as",
            defaultextension=".srt",
            filetypes=[("SubRip subtitle", "*.srt"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select a trained model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _append_log(self, text):
        self.log.config(state="normal")
        # Progress bars redraw themselves with carriage returns; emulate a
        # terminal by overwriting the last line.
        text = text.replace("\r\n", "\n")
        parts = text.split("\r")
        self.log.insert("end", parts[0])
        for part in parts[1:]:
            self.log.delete("end-1c linestart", "end-1c")
            self.log.insert("end", part)
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                self._append_log(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_run(self):
        if self._worker and self._worker.is_alive():
            return

        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("No file", "Please choose a video file first.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("Not found", f"File not found:\n{input_path}")
            return
        model_path = self.model_var.get().strip() or None
        if model_path and not os.path.isfile(model_path):
            messagebox.showerror("Not found", f"Model file not found:\n{model_path}")
            return

        try:
            args = SimpleNamespace(
                input=input_path,
                output=self.output_var.get().strip() or None,
                fps=float(self.fps_var.get()),
                crop=tuple(self._crop),
                model=model_path,
                threshold=float(self.threshold_var.get()),
                gap_tolerance=1,
                min_count=2,
                text=self.text_var.get() or "...",
            )
        except ValueError as exc:
            messagebox.showerror("Invalid option", f"Check numeric fields:\n{exc}")
            return

        self._cancel_event = threading.Event()
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.status_var.set("Working...")
        self._worker = threading.Thread(
            target=self._run_pipeline, args=(args, self._cancel_event), daemon=True
        )
        self._worker.start()

    def _on_cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.cancel_btn.config(state="disabled")
        self.status_var.set("Cancelling...")

    def _run_pipeline(self, args, cancel_event):
        """Runs on a background thread. stdout/stderr are redirected into the log."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        writer = _QueueWriter(self._log_queue, cancel_event)
        sys.stdout = sys.stderr = writer
        try:
            output_path = args.output or pipeline.default_output_path(args.input)
            blocks = pipeline.run_detection(args, should_cancel=cancel_event.is_set)
            print(f"Writing {len(blocks)} subtitle blocks to {output_path}...")
            write_srt(blocks, output_path)
            print("Done.\n")
            self.root.after(0, self._on_success, output_path, len(blocks))
        except pipeline.PipelineCancelled:
            writer.armed = False  # cancellation done; let the messages through
            print("\nCancelled.\n")
            self.root.after(0, self._on_cancelled)
        except Exception as exc:  # surface any failure in the log + a dialog
            writer.armed = False
            print(f"\nERROR: {exc}\n")
            self.root.after(0, self._on_error, str(exc))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.root.after(0, self._finish_run)

    def _finish_run(self):
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

    def _on_success(self, output_path, count):
        self.status_var.set(f"Done — {count} blocks written.")
        messagebox.showinfo("Finished", f"Wrote {count} subtitle blocks to:\n{output_path}")

    def _on_cancelled(self):
        self.status_var.set("Cancelled.")

    def _on_error(self, message):
        self.status_var.set("Error.")
        messagebox.showerror("Detection failed", message)


def main():
    root = tk.Tk()
    SubtitleTimerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
