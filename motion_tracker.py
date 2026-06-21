# -*- coding: utf-8 -*-
"""
Video Motion Tracker
====================

This script provides a utility for tracking the motion of objects in a video file.
It uses OpenCV's feature detection and optical flow algorithms. The user can
interactively select a region of interest (ROI), calibrate pixel-to-distance
measurements, and track feature points within that ROI across frames.

The application can be run in two modes:
1.  GUI Mode (default): A simple Tkinter interface allows the user to perform
    actions step-by-step.
2.  Auto Mode: A command-line driven mode to run a pre-defined sequence of
    operations (load, set ROI, set pixel size, track, save).

Usage:
    # Launch the GUI (default mode)
    python motion_tracker.py

    # Run the scripted sequence on a given video (no Tkinter required)
    python motion_tracker.py --mode auto --path data/Beehive_1.mp4
    python motion_tracker.py -m auto -v /path/to/video.mp4

Options:
    -m, --mode   manual (GUI) or auto (scripted sequence)  [default: manual]
    -v, --path   path to the video file (required for auto mode)

Dependencies:
- numpy
- opencv-python
- tkinter
- pandas
- tqdm
- matplotlib

@author: Ron Simenhois
"""

import numpy as np
import cv2
import pandas as pd
from tkinter import Tk, Button, simpledialog, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tqdm import tqdm, tqdm_gui
import matplotlib
matplotlib.use("Agg")  # headless-safe: speed plots are saved to PNG and overlaid, never shown
import matplotlib.pyplot as plt
import warnings
import argparse
import os

# Suppress minor warnings, e.g., from matplotlib or other libraries.
warnings.filterwarnings("ignore")


class VideoUtil:
    """
    A utility class to handle video processing for motion tracking.

    This class encapsulates all the functionalities related to loading a video,
    selecting a region of interest (ROI), calibrating scale, tracking features
    using Lucas-Kanade optical flow, and saving the results.
    """
    def __init__(self, video_file='', interactive=True):
        """
        Initializes the VideoUtil object.

        Args:
            video_file (str, optional): The path to the video file.
                                        If empty, a file dialog will open.
            interactive (bool): When True, user prompts (calibration values, save
                                paths, notices) use Tk dialogs. When False they
                                fall back to the console and sensible defaults, so
                                the tracker runs without Tkinter (e.g. 'auto' mode
                                or on systems with a broken Tk install).
        """
        self.interactive = interactive
        if not video_file:
            if not interactive:
                raise FileNotFoundError("No video file provided (required in non-interactive mode).")
            # Hide the root Tkinter window and open a file dialog.
            Tk().withdraw()
            self.video_file = askopenfilename(title="Select a video file")
            if not self.video_file:
                raise FileNotFoundError("No video file selected.")
        else:
            self.video_file = video_file

        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_file}")

        # --- Parameters for ShiTomasi corner detection ---
        self.feature_params = dict(maxCorners=20,
                                   qualityLevel=0.3,
                                   minDistance=15,
                                   blockSize=7)

        # --- Parameters for Lucas-Kanade optical flow ---
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # --- Instance variables ---
        self.pix_height = 1.0  # meters per pixel (vertical)
        self.pix_width = 1.0   # meters per pixel (horizontal)
        self.trak_video = []   # Buffer for tracked video frames
        self.frame_num = 0     # Current frame number for UI interactions
        self.drawing = False   # Flag for mouse drawing events
        self.stab_mask = None  # Optional user-marked static area for stabilization
        self.ix, self.iy = -1, -1 # Initial drawing coordinates

        # Pre-load video frames into memory.
        self.load_video()
        # Initialize a default mask covering the entire frame.
        self.mask = np.ones_like(self.video_buffer[0], dtype=np.uint8) * 255


    def load_video(self):
        """
        Loads the entire video file into a NumPy array in memory.

        This method reads all frames from the video source and stores them in
        `self.video_buffer`. This is memory-intensive but allows for fast,
        non-linear access to frames for interactive tasks like trimming.
        """
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.video_buffer = np.zeros((self.length, self.height, self.width, 3), dtype='uint8')
        print("Loading video into memory...")
        for i in tqdm(range(self.length), desc="Loading frames"):
            ret, frame = self.cap.read()
            if not ret:
                # If reading fails, truncate the buffer to the frames read.
                self.length = i
                self.video_buffer = self.video_buffer[:i]
                break
            self.video_buffer[i] = frame.copy()

        self.cap.release()
        self.cap = None
        print("Video loaded successfully.")


    def on_mouse(self, event, x, y, flags, param):
        """
        OpenCV mouse callback function to draw a rectangle.

        Used for selecting ROI and calibration areas.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Create a copy to draw the rectangle dynamically
                self.img_cp = self.img.copy()
                cv2.rectangle(self.img_cp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                cv2.rectangle(self.img_cp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                # Define upper-left and lower-right corners
                self.ul_cr = (min(x, self.ix), min(y, self.iy))
                self.lr_cr = (max(x, self.ix), max(y, self.iy))


    def set_roi(self):
        """
        Opens an interactive window to define the Region of Interest (ROI).

        The user can draw a rectangle on a frame to specify the area where
        features should be tracked. A trackbar allows scrubbing through frames.
        """
        self.drawing = False
        zoom = 1
        window_name = 'Draw rectangle around track area | - to zoom out | Esc to save'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Trackbar callback to change the displayed frame
        def _change_frame(trackbar_val):
            self.frame_num = trackbar_val
            self.img = self.video_buffer[self.frame_num].copy()
            self.img_cp = self.img.copy()

        cv2.createTrackbar("Frame #: ", window_name, 0, self.length - 1, _change_frame)

        # Initialize with the first frame
        self.frame_num = 0
        self.img = self.video_buffer[self.frame_num].copy()
        self.img_cp = self.img.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])

        while True:
            cv2.imshow(window_name, self.img_cp)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # ESC key
                break
            if k == ord('-'): # Zoom out
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_cp = self.img.copy()
                zoom *= 2 # Adjust zoom factor for coordinate mapping

        cv2.destroyAllWindows()
        # Create a new mask based on the user-drawn rectangle
        self.mask = np.zeros_like(self.video_buffer[0], dtype=np.uint8)
        # Apply zoom factor to map coordinates back to the original frame size
        top_y = self.ul_cr[1] * zoom
        bot_y = self.lr_cr[1] * zoom
        left_x = self.ul_cr[0] * zoom
        right_x = self.lr_cr[0] * zoom
        self.mask[top_y:bot_y, left_x:right_x] = 255


    def set_static_area(self):
        """
        Mark one or more *static* reference regions for camera-motion stabilization.

        By default the stabilizer assumes everything outside the tracking ROI is
        static background. That fails when the scene has moving distractions there
        (people, wind-blown trees, water). This lets the user draw rectangles over
        areas that are genuinely still (a building, the ground, a rock) so only
        those feed the camera-motion estimate.

        Controls: drag to draw a box, 'a' or Enter to add it, 'c' to clear all,
        '-' to zoom out, Esc to finish. Adding nothing leaves the default
        (inverse-of-ROI) behavior.
        """
        self.drawing = False
        zoom = 1
        window_name = 'Draw STATIC areas | drag, a/Enter to add | c clear | - zoom | Esc done'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

        stab = np.zeros(self.video_buffer[0].shape[:2], dtype=np.uint8)
        self.img = self.video_buffer[0].copy()
        self.img_cp = self.img.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (0, 0)

        while True:
            cv2.imshow(window_name, self.img_cp)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # Esc: finish
                break
            elif k in (ord('a'), 13):  # add the current rectangle to the mask
                top_y, bot_y = self.ul_cr[1] * zoom, self.lr_cr[1] * zoom
                left_x, right_x = self.ul_cr[0] * zoom, self.lr_cr[0] * zoom
                if bot_y > top_y and right_x > left_x:
                    stab[top_y:bot_y, left_x:right_x] = 255
                    # Persist the box on the base image so it stays visible while
                    # further boxes are drawn.
                    cv2.rectangle(self.img, self.ul_cr, self.lr_cr, (255, 0, 0), 2)
                    self.img_cp = self.img.copy()
            elif k == ord('c'):  # clear everything and reset zoom
                stab[:] = 0
                zoom = 1
                self.img = self.video_buffer[0].copy()
                self.img_cp = self.img.copy()
            elif k == ord('-'):  # zoom out
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_cp = self.img.copy()
                zoom *= 2

        cv2.destroyAllWindows()
        self.stab_mask = stab if stab.any() else None
        if self.stab_mask is not None:
            print(f"Static reference area set ({(stab > 0).mean() * 100:.0f}% of frame).")
        else:
            print("No static area selected; using default (everything outside the ROI).")


    def _prompt(self, message):
        """
        Ask the user for a string value. Uses a Tk dialog when interactive,
        otherwise falls back to a console prompt so no Tkinter is required.
        Returns the entered string, or None if blank/cancelled.
        """
        if self.interactive:
            Tk().withdraw()
            return simpledialog.askstring(title="Motion Tracker", prompt=message)
        try:
            resp = input(message + " ").strip()
        except EOFError:
            return None
        return resp or None

    def _notify(self, title, message):
        """Show an informational message (Tk dialog when interactive, else print)."""
        if self.interactive:
            messagebox.showinfo(title, message)
        else:
            print(f"{title}: {message}")

    def _save_path(self, default_path, filetypes, title):
        """
        Resolve a path to save to. Uses a Tk save dialog when interactive,
        otherwise auto-saves to ``default_path`` (no Tkinter required).
        """
        if self.interactive:
            Tk().withdraw()
            return asksaveasfilename(initialfile=default_path,
                                     defaultextension=os.path.splitext(default_path)[1],
                                     filetypes=filetypes, title=title)
        print(f"{title} -> {default_path}")
        return default_path

    def set_pxl_size(self):
        """
        Calibrates the pixel-to-meter ratio.

        The user draws a rectangle over an object of known real-world size.
        Dialog boxes prompt for the height and width in meters to calculate
        the conversion factors.
        """
        zoom = 1
        self.drawing = False
        window_name = 'Draw a rectangle with a known size (in meters) | Esc to save'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Initialize with first frame and default rectangle
        self.img = self.video_buffer[0].copy()
        self.img_cp = self.img.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])

        while True:
            cv2.imshow(window_name, self.img_cp)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # ESC key
                break
            if k == ord('-'): # Zoom out
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_cp = self.img.copy()
                zoom *= 2

        cv2.destroyAllWindows()
        # Calculate the pixel dimensions of the drawn rectangle
        height_px = abs(self.ul_cr[1] - self.lr_cr[1]) * zoom
        width_px = abs(self.ul_cr[0] - self.lr_cr[0]) * zoom

        # --- Get real-world height ---
        rect_height_m_str = self._prompt("Enter rectangle height in meters (blank if unknown):")
        try:
            self.pix_height = float(rect_height_m_str) / height_px
        except (ValueError, TypeError, ZeroDivisionError):
            self.pix_height = None

        # --- Get real-world width ---
        rect_width_m_str = self._prompt("Enter rectangle width in meters (blank if unknown):")
        try:
            self.pix_width = float(rect_width_m_str) / width_px
        except (ValueError, TypeError, ZeroDivisionError):
            self.pix_width = None

        # --- Handle cases where one dimension is unknown (assume square pixels) ---
        if self.pix_height is None and self.pix_width is None:
            print("Warning: No calibration data entered. Using 1.0 for pixel ratios.")
            self.pix_height, self.pix_width = 1.0, 1.0
        elif self.pix_height is None:
            self.pix_height = self.pix_width
        elif self.pix_width is None:
            self.pix_width = self.pix_height

        print(f"Calibration set: 1 pixel = {self.pix_width:.4f}m (width), {self.pix_height:.4f}m (height)")


    @staticmethod
    def _apply_affine(M, x, y):
        """Maps point (x, y) through a 2x3 affine transform M."""
        ex = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ey = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return ex, ey

    def _camera_transform(self, prev_pts, next_pts):
        """
        Estimates the camera (background) motion between two frames.

        Fits a partial-affine transform (translation + rotation + uniform scale)
        from the background feature correspondences using RANSAC. Falls back to a
        pure median translation when too few points are available, and to the
        identity transform when there are none.

        Args:
            prev_pts (np.ndarray): Background points in the previous frame, shape (N, 2).
            next_pts (np.ndarray): The same points in the current frame, shape (N, 2).

        Returns:
            tuple[np.ndarray, int, str]: the 2x3 transform mapping prev->next, the
            number of points that informed it, and the method used
            ('affine', 'translation', or 'none').
        """
        n = 0 if prev_pts is None else len(prev_pts)
        if n >= 3:
            M, inliers = cv2.estimateAffinePartial2D(
                prev_pts, next_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
            if M is not None:
                n_used = int(inliers.sum()) if inliers is not None else n
                return M, n_used, 'affine'
        if n >= 1:
            # Fallback: median translation handles pan/tilt but not rotation/zoom.
            d = (next_pts - prev_pts).reshape(-1, 2)
            tx, ty = np.median(d[:, 0]), np.median(d[:, 1])
            return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float64), n, 'translation'
        # No background information: identity (no correction this frame).
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), 0, 'none'

    def track(self, draw_speeds=True, stabilize=True):
        """
        Performs the motion tracking using Lucas-Kanade optical flow.

        Detects features in the first frame within the ROI and tracks their
        movement in subsequent frames. It calculates the displacement and speed,
        saves the data to a pandas DataFrame, and generates a video with
        tracking overlays.

        When ``stabilize`` is True, features are also tracked in the background
        (outside the ROI) to estimate the camera's own motion, which is then
        subtracted from the ROI motion. This removes apparent motion caused by a
        panning/handheld camera, leaving the object's motion relative to the scene.

        Args:
            draw_speeds (bool): If True, generates and overlays a speed plot
                                on the output video.
            stabilize (bool): If True, compensate for camera motion using
                              background features.
        """
        self.trak_video = []
        if self.interactive and messagebox.askyesno("Frame Range", "Do you want to track a specific range of frames?"):
            start_frame, end_frame = self.trim_video(trim=False)
        else:
            start_frame, end_frame = 0, self.length

        # --- Initialization for Tracking ---
        color = np.random.randint(0, 255, (100, 3))
        old_frame = self.video_buffer[start_frame]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Use the grayscale version of the ROI mask for feature detection
        mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_gray, **self.feature_params)

        if p0 is None:
            self._notify("Tracking Error", "No features found in the selected ROI.")
            return

        # --- Background features for camera-motion compensation ---
        bg_mask = None
        bg_p0 = None
        bg_params = dict(self.feature_params, maxCorners=200)
        if stabilize:
            # Always exclude the tracked object (dilated ROI, to avoid edge bleed)
            # so it cannot contaminate the camera-motion estimate.
            roi_dilated = cv2.dilate(mask_gray, np.ones((15, 15), np.uint8))
            if self.stab_mask is not None:
                # User marked explicit static area(s): draw reference features
                # only from there (minus any overlap with the object).
                bg_mask = cv2.bitwise_and(self.stab_mask, cv2.bitwise_not(roi_dilated))
            else:
                # Default: treat everything outside the ROI as static background.
                bg_mask = cv2.bitwise_not(roi_dilated)
            bg_p0 = cv2.goodFeaturesToTrack(old_gray, mask=bg_mask, **bg_params)
            if bg_p0 is None:
                print("Warning: no background features found; stabilization disabled.")
                stabilize = False

        # Create DataFrame to store motion data (raw, plus corrected when stabilizing)
        cols = []
        for i in range(self.feature_params['maxCorners']):
            cols += [f'p{i}_x', f'p{i}_y']
            if stabilize:
                cols += [f'p{i}_x_corr', f'p{i}_y_corr']
        if stabilize:
            cols += ['bg_points', 'cam_dx', 'cam_dy']
        motion_df = pd.DataFrame(columns=cols)

        overlay_mask = np.zeros_like(old_frame) # Mask for drawing tracking lines
        speeds_x, speeds_y = [], []
        
        print("Tracking motion...")
        for idx in tqdm(range(start_frame, end_frame), desc="Tracking frames"):
            frame = self.video_buffer[idx]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Calculate Optical Flow ---
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

            # --- Select and process good points ---
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            else:
                good_new, good_old = [], []

            # --- Estimate camera motion from background features ---
            cam_M, n_bg, cam_method = None, 0, 'none'
            if stabilize and bg_p0 is not None and len(bg_p0) > 0:
                bg_p1, bg_st, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, bg_p0, None, **self.lk_params)
                if bg_p1 is not None:
                    bg_good_new = bg_p1[bg_st == 1]
                    bg_good_old = bg_p0[bg_st == 1]
                    cam_M, n_bg, cam_method = self._camera_transform(bg_good_old, bg_good_new)
                    bg_p0 = bg_good_new.reshape(-1, 1, 2)

            frame_motion_data = {}
            dxdt, dydt = [], []          # raw speeds (m/s)
            cdxdt, cdydt = [], []        # camera-corrected speeds (m/s)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                # Store raw displacement in real-world units (meters).
                # Y is negated so that up is positive (image rows increase
                # downward, so screen-down is d->b positive without the flip).
                frame_motion_data[f'p{i}_x'] = (a - c) * self.pix_width
                frame_motion_data[f'p{i}_y'] = (d - b) * self.pix_height
                dxdt.append((a - c) * self.pix_width * self.fps)
                dydt.append((d - b) * self.pix_height * self.fps)

                # Subtract camera motion: where would a static point at (c, d)
                # land under the estimated camera transform? The residual is the
                # object's true motion relative to the scene. Y negated (up positive).
                if cam_M is not None:
                    ex, ey = self._apply_affine(cam_M, c, d)
                    corr_dx = (a - ex) * self.pix_width
                    corr_dy = (ey - b) * self.pix_height
                    frame_motion_data[f'p{i}_x_corr'] = corr_dx
                    frame_motion_data[f'p{i}_y_corr'] = corr_dy
                    cdxdt.append(corr_dx * self.fps)
                    cdydt.append(corr_dy * self.fps)

                # Draw tracking lines and points
                overlay_mask = cv2.line(overlay_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            if cam_M is not None:
                # Diagnostic: median background displacement in meters this frame.
                frame_motion_data['bg_points'] = n_bg
                frame_motion_data['cam_dx'] = float(np.median(cdxdt) - np.median(dxdt)) / self.fps if dxdt else 0.0
                frame_motion_data['cam_dy'] = float(np.median(cdydt) - np.median(dydt)) / self.fps if dydt else 0.0

            motion_df = pd.concat([motion_df, pd.DataFrame([frame_motion_data])], ignore_index=True)
            img = cv2.add(frame, overlay_mask)

            # --- Calculate and Display Median Speed ---
            # When stabilizing, report the corrected speed; otherwise the raw speed.
            use_corr = cam_M is not None and cdxdt
            median_dxdt = np.median(cdxdt if use_corr else dxdt) if (cdxdt or dxdt) else 0
            median_dydt = np.median(cdydt if use_corr else dydt) if (cdydt or dydt) else 0
            speeds_x.append(abs(median_dxdt))
            speeds_y.append(abs(median_dydt))

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_y1 = int(0.10 * self.height)
            text_y2 = int(0.15 * self.height)
            cv2.putText(img, f"dx/dt = {median_dxdt:.2f} m/s", (10, text_y1), font, 1, (0, 0, 255), 2)
            cv2.putText(img, f"dy/dt = {median_dydt:.2f} m/s", (10, text_y2), font, 1, (0, 0, 255), 2)
            if stabilize:
                text_y3 = int(0.20 * self.height)
                conf_color = (0, 200, 0) if n_bg >= 10 else (0, 165, 255)
                cv2.putText(img, f"bg pts: {n_bg} ({cam_method})", (10, text_y3),
                            font, 0.7, conf_color, 2)

            self.trak_video.append(img)

            # --- Update for next iteration ---
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # Re-detect background features when too many have been lost.
            if stabilize and (bg_p0 is None or len(bg_p0) < 10):
                bg_p0 = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **bg_params)

        # --- Optional: Overlay Speed Plots ---
        if draw_speeds and self.trak_video:
            self.overlay_speed_plots(speeds_x, speeds_y, start_frame, end_frame)
        
        cv2.destroyAllWindows()
        print("Tracking complete.")

        # --- Save Motion Data to CSV ---
        initial_filename = f"{os.path.splitext(self.video_file)[0]}_track_frames_{start_frame}_to_{end_frame}.csv"
        save_filename = self._save_path(initial_filename,
                                        [("CSV (Comma-separated)", "*.csv"), ("All Files", "*.*")],
                                        "Save Motion Data")
        if save_filename:
            motion_df.to_csv(save_filename, index=False)
            print(f"Motion data saved to {save_filename}")


    def overlay_speed_plots(self, speeds_x, speeds_y, start, end):
        """Helper function to generate and overlay speed plots on video frames."""
        print("Generating speed plots for video overlay...")
        plt.ioff() # Turn off interactive plotting
        
        # Define plot dimensions and limits
        ch_h, ch_w = int(self.height / 2), int(self.width / 5)
        xlim = [0, end - start]
        xspdlim = [0, max(0.1, max(speeds_x) * 1.1)]
        yspdlim = [0, max(0.1, max(speeds_y) * 1.1)]

        temp_chart_path = 'spd_chart.png'

        for i, img in enumerate(tqdm(self.trak_video, desc="Overlaying plots")):
            fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(ch_w/100, ch_h/100), dpi=100)
            
            ax[0].plot(range(i + 1), speeds_x[:i + 1], 'r.-')
            ax[0].set_title('X-axis speed (m/s)', fontsize=10)
            ax[0].set_xlim(xlim)
            ax[0].set_ylim(xspdlim)
            
            ax[1].plot(range(i + 1), speeds_y[:i + 1], 'r.-')
            ax[1].set_title('Y-axis speed (m/s)', fontsize=10)
            ax[1].set_xlabel('Frame #', fontsize=8)
            ax[1].set_xlim(xlim)
            ax[1].set_ylim(yspdlim)
            
            plt.tight_layout()
            fig.savefig(temp_chart_path, transparent=True)
            plt.close(fig)

            # Load chart and overlay onto the video frame
            chart = cv2.imread(temp_chart_path, cv2.IMREAD_UNCHANGED)
            chart_resized = cv2.resize(chart, (ch_w, ch_h), interpolation=cv2.INTER_AREA)
            
            roi = img[:ch_h, -ch_w:]
            alpha_mask = chart_resized[:, :, 3] / 255.0
            alpha_mask_3d = np.stack([alpha_mask] * 3, axis=-1)
            
            chart_rgb = chart_resized[:, :, :3]
            blended_roi = (chart_rgb * alpha_mask_3d + roi * (1 - alpha_mask_3d)).astype(np.uint8)
            img[:ch_h, -ch_w:] = blended_roi
            self.trak_video[i] = img # Update the frame in the buffer

            cv2.imshow('Tracking with Speed Plot', img)
            cv2.waitKey(10) # Small delay to show progress
            
        cv2.destroyAllWindows()
        if os.path.exists(temp_chart_path):
            os.remove(temp_chart_path)
        plt.ion() # Turn interactive plotting back on


    def save_tracking_video(self, fps=10):
        """
        Saves the tracked video (with overlays) to a file.

        Args:
            fps (int): The frames per second for the output video.
        """
        if not self.trak_video:
            self._notify("No Video", "No tracking video to save. Please run tracking first.")
            return

        initial_filename = f"{os.path.splitext(self.video_file)[0]}_tracked.mp4"
        save_file = self._save_path(initial_filename,
                                    [("MP4 video file", "*.mp4"), ("All files", "*.*")],
                                    "Save Tracked Video")
        if not save_file:
            print("Video saving cancelled.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps,
                              frameSize=(self.width, self.height), isColor=True)

        for frame in tqdm(self.trak_video, desc="Saving video"):
            out.write(frame)

        out.release()
        print(f"Tracked video saved successfully to {save_file}")


    @staticmethod
    def _grid_points(mask_gray, spacing, max_points):
        """Seed a regular grid of points (pixels) over the nonzero ROI mask."""
        ys, xs = np.where(mask_gray > 0)
        if xs.size == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        gy, gx = np.mgrid[y0:y1 + 1:spacing, x0:x1 + 1:spacing]
        pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
        inside = mask_gray[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0
        pts = pts[inside]
        if pts.shape[0] > max_points:                # thin evenly to the cap
            pts = pts[np.linspace(0, pts.shape[0] - 1, max_points).astype(int)]
        return pts.reshape(-1, 1, 2)

    def track_markers(self, start=0, end=None, n_markers=100,
                      quality_level=0.05, min_distance=8, stabilize=False,
                      seed='features', grid_spacing=25):
        """
        Track feature points inside the ROI and return their absolute trajectories.

        Unlike track() (which is built for visualization and stores frame-to-frame
        deltas, dropping/renumbering points as optical flow loses them), this keeps
        a *stable* marker identity: a point that is lost becomes NaN from then on
        and is never renumbered, so a trajectory's column index is a fixed marker
        ID. Intended as the data source for downstream analysis (e.g. PST metrics).

        Args:
            start, end (int): frame range to track (end defaults to full length).
            n_markers (int): maximum number of features to detect in the ROI.
            quality_level (float): Shi-Tomasi quality threshold for detection.
                              Lower than track()'s default (0.05 vs 0.3) so many
                              markers are found spread across the column.
            min_distance (int): minimum pixel spacing between detected markers.
            seed (str): 'features' to detect Shi-Tomasi corners, or 'grid' to seed
                        a regular grid of points across the ROI (better for
                        low-texture surfaces like snow, which yield few corners).
            grid_spacing (int): pixel spacing of the grid when seed='grid'.
            stabilize (bool): if True, subtract camera motion (estimated from
                              background features and integrated frame to frame) so
                              trajectories are relative to the scene.

        Returns:
            dict with:
              'positions'  : (n_frames, n_markers, 2) float32 array of (x, y) pixel
                             positions; NaN once a marker is lost.
              'times'      : (n_frames,) array of seconds from `start`.
              'pix_width'  : meters per pixel, horizontal.
              'pix_height' : meters per pixel, vertical.
              'fps'        : frames per second.
              'start','end': the frame range tracked.
              'frame0'     : the first frame (BGR) for reference/plotting.
        """
        end = self.length if end is None else min(end, self.length)
        n_frames = end - start
        if n_frames < 2:
            raise ValueError("track_markers needs at least two frames.")

        block_size = self.feature_params.get('blockSize', 7)
        detect_params = dict(maxCorners=int(n_markers), qualityLevel=float(quality_level),
                             minDistance=int(min_distance), blockSize=block_size)
        bg_detect_params = dict(detect_params, maxCorners=200)
        mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        old_frame = self.video_buffer[start]
        prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        if seed == 'grid':
            p0 = self._grid_points(mask_gray, grid_spacing, int(n_markers))
            if p0 is None:
                raise RuntimeError("ROI is empty - cannot seed a grid.")
        else:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask_gray, **detect_params)
            if p0 is None:
                raise RuntimeError("No features found in the ROI to track.")
        n = len(p0)
        print(f"Seeding {n} markers ({seed}) in the ROI.")
        if seed == 'features' and n < 30:
            print(f"  note: only {n} corners found - the surface may be low-texture. "
                  f"Try --seed grid, or lower --quality / --min-distance for more points.")

        # raw  : pixel positions used to feed optical flow (LK continuity)
        # pos  : analysis positions (camera-stabilized when requested)
        raw = np.full((n_frames, n, 2), np.nan, dtype=np.float32)
        pos = np.full((n_frames, n, 2), np.nan, dtype=np.float32)
        raw[0] = p0.reshape(n, 2)
        pos[0] = p0.reshape(n, 2)
        alive = np.ones(n, dtype=bool)

        # Optional background features for camera-motion compensation.
        bg_mask = None
        bg_p0 = None
        if stabilize:
            roi_dilated = cv2.dilate(mask_gray, np.ones((15, 15), np.uint8))
            bg_mask = cv2.bitwise_not(roi_dilated)
            bg_p0 = cv2.goodFeaturesToTrack(prev_gray, mask=bg_mask, **bg_detect_params)
            if bg_p0 is None:
                print("Warning: no background features; trajectories will not be camera-stabilized.")
                stabilize = False

        print("Tracking marker trajectories...")
        for f in tqdm(range(1, n_frames), desc="Tracking markers"):
            frame_gray = cv2.cvtColor(self.video_buffer[start + f], cv2.COLOR_BGR2GRAY)

            # Estimate camera motion (prev -> current) from background points.
            cam_M = None
            if stabilize and bg_p0 is not None and len(bg_p0) > 0:
                bg_new, bg_st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, bg_p0, None, **self.lk_params)
                if bg_new is not None:
                    sel = bg_st.ravel() == 1
                    cam_M, _, _ = self._camera_transform(bg_p0[sel], bg_new[sel])
                    bg_p0 = bg_new[sel].reshape(-1, 1, 2)
                if bg_p0 is None or len(bg_p0) < 10:
                    bg_p0 = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **bg_detect_params)

            idx = np.where(alive)[0]
            if len(idx) == 0:
                break
            pts_in = raw[f - 1, idx].reshape(-1, 1, 2)
            new, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, pts_in, None, **self.lk_params)
            st = st.ravel() == 1
            new = new.reshape(-1, 2)
            for k, gi in enumerate(idx):
                if not st[k]:
                    alive[gi] = False  # leaves the rest of this trajectory as NaN
                    continue
                a, b = new[k]
                raw[f, gi] = (a, b)
                if stabilize:
                    c, d = raw[f - 1, gi]
                    ex, ey = self._apply_affine(cam_M, c, d) if cam_M is not None else (c, d)
                    px, py = pos[f - 1, gi]
                    pos[f, gi] = (px + (a - ex), py + (b - ey))
                else:
                    pos[f, gi] = (a, b)

            prev_gray = frame_gray

        return {
            'positions': pos,
            'times': np.arange(n_frames) / float(self.fps),
            'pix_width': self.pix_width,
            'pix_height': self.pix_height,
            'fps': float(self.fps),
            'start': start,
            'end': end,
            'frame0': old_frame.copy(),
        }


    def trim_video(self, trim):
        """
        Provides an interactive UI to select a start and end frame.

        Args:
            trim (bool): If True, the video buffer will be permanently trimmed
                         to the selected range. If False, it only returns the
                         selected range without modifying the buffer.

        Returns:
            tuple[int, int]: The selected start and end frame numbers.
        """
        def _move(frame_n):
            self.frame_num = frame_n

        win_name = "Trim Video | s: start, e: end, f: fwd, b: back, Esc: exit"
        cv2.namedWindow(win_name)
        start, end = 0, self.length - 1
        cv2.createTrackbar("Frame #:", win_name, 0, self.length - 1, _move)

        while True:
            frame = self.video_buffer[self.frame_num]
            display_frame = frame.copy()
            cv2.putText(display_frame, f'Start: {start} | End: {end}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(win_name, display_frame)

            k = cv2.waitKey(30) & 0xFF
            if k == ord("s"): start = self.frame_num
            elif k == ord("e"): end = self.frame_num
            elif k == ord("f"): self.frame_num = min(self.frame_num + 1, self.length - 1)
            elif k == ord("b"): self.frame_num = max(0, self.frame_num - 1)
            elif k == 27: break
            cv2.setTrackbarPos("Frame #:", win_name, self.frame_num)
        
        cv2.destroyAllWindows()
        # Ensure start is before end
        start, end = min(start, end), max(start, end)

        if trim:
            self.video_buffer = self.video_buffer[start:end, ...]
            self.length = self.video_buffer.shape[0]
            print(f"Video trimmed to frames {start} through {end}. New length: {self.length} frames.")

        return start, end


    def play_video(self):
        """Plays the video currently loaded in the buffer with a trackbar."""
        if not hasattr(self, 'video_buffer') or self.length == 0:
            messagebox.showerror("Error", "No video loaded.")
            return

        def _move(frame_n):
            self.frame_num = frame_n

        win_name = f"Play Video: {os.path.basename(self.video_file)}"
        cv2.namedWindow(win_name)
        cv2.createTrackbar("Frame #:", win_name, 0, self.length - 1, _move)
        
        self.frame_num = 0
        while True:
            if self.frame_num >= self.length: break
            
            cv2.setTrackbarPos("Frame #:", win_name, self.frame_num)
            cv2.imshow(win_name, self.video_buffer[self.frame_num])
            
            k = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            if k == 27: break
            
            self.frame_num += 1
            
        cv2.destroyAllWindows()


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
