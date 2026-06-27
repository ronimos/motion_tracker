# -*- coding: utf-8 -*-
"""
Video Motion Tracker - app
==========================

Tkinter GUI and command-line entry point for the video tracker. The tracking
engine itself lives in ``video_tracker.VideoUtil``; this module wires it to a
simple button GUI (manual mode) and a scripted sequence (auto mode).

``VideoUtil`` is re-exported here for backward compatibility, so existing
``from motion_tracker import VideoUtil`` imports keep working.

Usage:
    # Launch the GUI (default mode)
    python motion_tracker.py

    # Run the scripted sequence on a given video (no Tkinter required)
    python motion_tracker.py --mode auto --path data/Beehive_1.mp4

@author: Ron Simenhois
"""

import argparse
from tkinter import Tk, Button, messagebox

from video_tracker import VideoUtil


class Gui:
    """A simple Tkinter GUI to interact with the VideoUtil class."""
    def __init__(self):
        """Initializes the GUI window and its widgets."""
        self.video_loaded = False
        self.video_util = None

        self.window = Tk()
        self.window.wm_title('Video Motion Tracker')

        # --- GUI Buttons ---
        btn_width = 25
        Button(self.window, text='Load Video', command=self.load_video, width=btn_width).grid(row=1, column=1, padx=5, pady=2)
        Button(self.window, text='Set Area to Track (ROI)', command=self.get_roi, width=btn_width).grid(row=2, column=1, padx=5, pady=2)
        Button(self.window, text='Set Static Area (optional)', command=self.set_static_area, width=btn_width).grid(row=3, column=1, padx=5, pady=2)
        Button(self.window, text='Set Pixel Size (Calibration)', command=self.set_pxl_size, width=btn_width).grid(row=4, column=1, padx=5, pady=2)
        Button(self.window, text='Track Motion', command=self.video_track, width=btn_width).grid(row=5, column=1, padx=5, pady=2)
        Button(self.window, text='Trim Video', command=self.trim_video, width=btn_width).grid(row=6, column=1, padx=5, pady=2)
        Button(self.window, text='Save Tracked Video', command=self.save_tracked, width=btn_width).grid(row=7, column=1, padx=5, pady=2)
        Button(self.window, text='Play Original Video', command=self.play_video, width=btn_width).grid(row=8, column=1, padx=5, pady=2)
        Button(self.window, text='Exit', command=self.quit, width=btn_width).grid(row=9, column=1, padx=5, pady=5)

        self.window.mainloop()

    def _check_video_loaded(self):
        """Helper to check if a video is loaded before calling methods."""
        if not self.video_loaded:
            messagebox.showerror('No Video Selected', 'Please load a video first.')
            return False
        return True

    def load_video(self):
        """Loads a video using VideoUtil."""
        try:
            self.video_util = VideoUtil()
            self.video_loaded = True
            messagebox.showinfo("Success", "Video loaded successfully.")
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("Error", str(e))
            self.video_loaded = False

    def get_roi(self):
        """Sets the tracking ROI."""
        if self._check_video_loaded(): self.video_util.set_roi()

    def set_static_area(self):
        """Marks static reference area(s) for stabilization."""
        if self._check_video_loaded(): self.video_util.set_static_area()

    def set_pxl_size(self):
        """Sets the pixel size for calibration."""
        if self._check_video_loaded(): self.video_util.set_pxl_size()

    def video_track(self):
        """Runs the motion tracking."""
        if self._check_video_loaded(): self.video_util.track()

    def save_tracked(self):
        """Saves the generated tracked video."""
        if self._check_video_loaded(): self.video_util.save_tracking_video(fps=30)

    def trim_video(self):
        """Trims the video buffer."""
        if self._check_video_loaded(): self.video_util.trim_video(trim=True)

    def play_video(self):
        """Plays the original video."""
        if self._check_video_loaded(): self.video_util.play_video()

    def quit(self):
        """Destroys the GUI window and exits."""
        self.window.destroy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Video Motion Tracking Utility")
    ap.add_argument("-m", "--mode", default="manual", choices=["manual", "auto"],
                    help="Execution mode: 'manual' (GUI) or 'auto' (scripted).")
    ap.add_argument("-v", "--path", default="", help="Path to the video file for 'auto' mode.")
    args = ap.parse_args()

    if args.mode == "auto":
        if not args.path:
            print("Error: For 'auto' mode, you must provide a video path using -v or --path.")
        else:
            try:
                # ROI/calibration still use OpenCV windows (mouse), but all
                # text prompts and save paths fall back to the console so this
                # mode needs no Tkinter.
                print("--- Running in Automated Sequence Mode ---")
                v = VideoUtil(args.path, interactive=False)
                print("Step 1: Set Region of Interest (ROI)")
                v.set_roi()
                print("Step 2: (optional) Set a static area for stabilization")
                if input("Draw a static reference area? [y/N] ").strip().lower().startswith("y"):
                    v.set_static_area()
                print("Step 3: Set Pixel Size for Calibration")
                v.set_pxl_size()
                print("Step 4: Track Motion")
                v.track(draw_speeds=True)
                print("Step 5: Save Tracked Video")
                v.save_tracking_video(fps=10)
                print("--- Automated Sequence Complete ---")
            except Exception as e:
                print(f"An error occurred during auto mode: {e}")
    else:
        # Default mode: launch the Tkinter GUI
        Gui()
