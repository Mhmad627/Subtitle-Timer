"""Subtitle Timer & Extractor — simple desktop GUI.

A Tkinter front-end over the same pipeline used by main.py. Pick a video,
choose options, and click Extract. Progress/log output appears in the window;
the extraction runs on a background thread so the UI stays responsive.

Run directly:   python gui.py
Or build an exe: see build_exe.py / README.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

from output.srt_writer import write_srt
import main as pipeline


class _QueueWriter:
    """File-like object that forwards writes to a queue (thread -> GUI)."""

    def __init__(self, q):
        self._q = q

    def write(self, text):
        if text:
            self._q.put(text)

    def flush(self):
        pass


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        root.title("Subtitle Timer & Extractor")
        root.geometry("680x560")
        root.minsize(560, 460)

        self._log_queue = queue.Queue()
        self._worker = None

        self._build_widgets()
        self._poll_log_queue()

    # ---------- layout ----------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        # Input file row
        file_frame = ttk.LabelFrame(self.root, text="Video file")
        file_frame.pack(fill="x", **pad)

        self.input_var = tk.StringVar()
        entry = ttk.Entry(file_frame, textvariable=self.input_var)
        entry.pack(side="left", fill="x", expand=True, padx=6, pady=6)
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

        # Options
        opt = ttk.LabelFrame(self.root, text="Options")
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="Mode:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        self.mode_var = tk.StringVar(value="auto")
        ttk.Combobox(
            opt, textvariable=self.mode_var, values=["auto", "ocr", "audio"],
            state="readonly", width=10,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(opt, text="Whisper model:").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        self.model_var = tk.StringVar(value="base")
        ttk.Combobox(
            opt, textvariable=self.model_var,
            values=["tiny", "base", "small", "medium"],
            state="readonly", width=10,
        ).grid(row=0, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(opt, text="Sample fps:").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self.fps_var = tk.StringVar(value="2")
        ttk.Entry(opt, textvariable=self.fps_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(opt, text="Bottom crop:").grid(row=1, column=2, sticky="e", padx=6, pady=4)
        self.region_var = tk.StringVar(value="0.25")
        ttk.Entry(opt, textvariable=self.region_var, width=8).grid(
            row=1, column=3, sticky="w", padx=6, pady=4
        )

        ttk.Label(opt, text="Min subtitles:").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        self.minsub_var = tk.StringVar(value="5")
        ttk.Entry(opt, textvariable=self.minsub_var, width=8).grid(
            row=2, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(opt, text="OCR languages:").grid(row=2, column=2, sticky="e", padx=6, pady=4)
        self.lang_var = tk.StringVar(value="en")
        ttk.Entry(opt, textvariable=self.lang_var, width=10).grid(
            row=2, column=3, sticky="w", padx=6, pady=4
        )

        # Run button + status
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_frame, text="Extract Subtitles", command=self._on_run)
        self.run_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=10)

        # Log output
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, wrap="word", height=12, state="disabled")
        self.log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side="right", fill="y", pady=6)
        self.log.config(yscrollcommand=scroll.set)

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

    def _append_log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
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

        try:
            args = SimpleNamespace(
                input=input_path,
                output=self.output_var.get().strip() or None,
                mode=self.mode_var.get(),
                whisper_model=self.model_var.get(),
                fps=float(self.fps_var.get()),
                region=float(self.region_var.get()),
                min_subtitles=int(self.minsub_var.get()),
                languages=self.lang_var.get().split(),
                similarity=85.0,
                gpu=False,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid option", f"Check numeric fields:\n{exc}")
            return

        self.run_btn.config(state="disabled")
        self.status_var.set("Working...")
        self._worker = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
        self._worker.start()

    def _run_pipeline(self, args):
        """Runs on a background thread. stdout is redirected into the log."""
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter(self._log_queue)
        try:
            output_path = args.output or pipeline.default_output_path(args.input)
            blocks = pipeline.select_blocks(args)
            print(f"Writing {len(blocks)} subtitle blocks to {output_path}...")
            write_srt(blocks, output_path)
            print("Done.\n")
            self.root.after(0, self._on_success, output_path, len(blocks))
        except Exception as exc:  # surface any failure in the log + a dialog
            print(f"\nERROR: {exc}\n")
            self.root.after(0, self._on_error, str(exc))
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.run_btn.config(state="normal"))

    def _on_success(self, output_path, count):
        self.status_var.set(f"Done — {count} blocks written.")
        messagebox.showinfo("Finished", f"Wrote {count} subtitle blocks to:\n{output_path}")

    def _on_error(self, message):
        self.status_var.set("Error.")
        messagebox.showerror("Extraction failed", message)


def main():
    root = tk.Tk()
    SubtitleExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
