# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:10:16 2018

@author: Ron Simenhois
"""

import numpy as np
import cv2
from tkinter import Tk, simpledialog, Button, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
from tqdm import tqdm_gui
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

'''
Video_utill Class: measure motion of items throughout a video
'''

class Video_utill:
    def __init__(self, video_file=''):
            
        if video_file=='':
            Tk().withdraw()
            self.video_file = askopenfilename()
            self.cap = cv2.VideoCapture(self.video_file)
        else:
            self.video_file = video_file
            self.cap = cv2.VideoCapture(self.video_file)
            # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 20,
                                   qualityLevel = .3,
                                   minDistance = 10,
                                   blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.pxl_size = -1
        self.load_video()
        self.mask = self.video_buffer[0].copy()
        self.pix_heigth = 1
        self.pix_width = 1
        self.track_video = []
        self.frame_num = 0
        
    def load_video(self):
        """
        Read the video from the disc and load it into a numpy array
        ---------------------------------------------------------
        Params:
            self
        return:
            None
        """
        
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_buffer = np.zeros((self.length, self.height, self.width, 3), dtype='uint8')
        for i in tqdm_gui(range(self.length), leave=True):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.video_buffer[i]=frame.copy()
        plt.close()
        self.cap.release()
        self.cap = None
        return

    def _on_mouse(self, event, x, y, flag, param):
        """
        An internal function that gets called on mouse status chage event on a given window
        ---------------------------------------------------------
        Params:
            self
            event: (int) - mouse status change code
            x, y: (int) - mouse location in the active window
            flag: (int) - unused
            parram (dict) - unused
        return:
            None
        """      
        if self.drawing:
            self.img_copy = self.img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.rectangle(self.img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            else:
                self.img_show = self.img_copy.copy()
            heigth, width = self.img_copy.shape[:2]
            cv2.line(self.img_show, (0,y), (width, y), (0,0,0), 1)
            cv2.line(self.img_show, (x,0), (x, heigth), (0,0,0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                cv2.rectangle(self.img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                self.ul_cr = (max(min(x, self.ix), 0), max(min(y, self.iy), 0))
                self.lr_cr = (min(max(x, self.ix), self.img_copy.shape[1]), min(max(y, self.iy), self.img_copy.shape[0]))
        if self.drawing:
            self.img_show = cv2.addWeighted(self.img_show, 0.5, self.img_copy, 0.5, 0)
        return
  
    
    def set_roi(self):
        """
        Utility function to set area to track
        ---------------------------------------------------------
        Params:
            self
        return:
            None
        """
        def _change_frame(trackbar_val):
            self.frame_num = trackbar_val
            self.img = self.video_buffer[self.frame_num].copy()
            self.img_copy = self.img.copy()
            self.img_show = self.img_copy.copy()
            
        self.mask = np.zeros_like(self.video_buffer[0])
        zoom = 1
        self.drawing = False
        window_name = 'Draw a rectangle with around the area you want to track, Click "-" to resize, Esc to exit'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)
        cv2.createTrackbar("Frame #: ", window_name, 0, self.length, _change_frame)
        self.frame_num=0        
        self.img = self.video_buffer[self.frame_num].copy()
        self.img_copy = self.img.copy()
        self.img_show = self.img_copy.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])
        while True:
            cv2.imshow(window_name, self.img_show)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            if k == ord('-'):
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_copy = self.img.copy()
                self.img_show = self.img_copy.copy()
                self.ul_cr = (0, 0)
                self.lr_cr = (self.img.shape[1], self.img.shape[0])
                zoom *= 2
        cv2.destroyAllWindows()
        self.mask[self.ul_cr[1]*zoom:self.lr_cr[1]*zoom, self.ul_cr[0]*zoom:self.lr_cr[0]*zoom, :] = self.video_buffer[0,self.ul_cr[1]*zoom:self.lr_cr[1]*zoom, self.ul_cr[0]*zoom:self.lr_cr[0]*zoom, :]
        return
    
    
    def set_pxl_size(self):
        """
        Utility function to calculate pixle size in cm
        ---------------------------------------------------------
        Params:
            self
        return:
            None
        """
        zoom = 1
        self.drawing = False
        window_name = 'Draw a rectangle with a known height (in cm), Click "-" to resize, Esc to exit'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)
        self.img = self.video_buffer[0].copy()
        self.img_copy = self.img.copy()
        self.img_show = self.img_copy.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])
        while True:
            cv2.imshow(window_name, self.img_show)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            if k == ord('-'):
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_copy = self.img.copy()
                self.img_show = self.img_copy.copy()
                self.ul_cr = (0, 0)
                self.lr_cr = (self.img.shape[1], self.img.shape[0])
                zoom *= 2
        cv2.destroyAllWindows()
        heigth = np.abs(self.ul_cr[1] - self.lr_cr[1]) * zoom
        width = np.abs(self.ul_cr[0] - self.lr_cr[0]) * zoom
        Tk().withdraw()
        
        rect_height = simpledialog.askstring(title = "Get pixel size", 
                                             prompt = "Entire retcangle heigth in cm", 
                                             initialvalue="Click cancel if unknown")
        try:
            rect_height = float(rect_height)
            self.pix_heigth = rect_height/heigth
        except (TypeError, ValueError):
            rect_height="cancel"
            
        Tk().withdraw()
        rect_width = simpledialog.askstring(title = "Get pixel size", 
                                             prompt = "Entire retcangle width in cm", 
                                             initialvalue="Click cancel if unknown")
        
        try:
            rect_width = float(rect_width)
            self.pix_width = rect_width/width
        except (TypeError, ValueError):
            rect_height="cancel"
            
        if rect_height=="cancel":
            self.pix_heigth = self.pix_width
        elif rect_width=="cancel":
            self.width = self.pix_heigth
        return
            
        
    def track(self):
        """
        Utility function to track motion over frames in a video.
        This function uses Shi-Tomasi corner detector to track and 
        Lucas-Kanade to track corner morion over the video's frames.
        The corners morions are saved in csv file.
        ---------------------------------------------------------
        Params:
            self
        return:
            None
        """
        if messagebox.askyesno("", "Do you want to set frames to track?"):
            start, end = self.trim_video(trim=False)
        else:
            start = 0
            end = self.length
        # Create DataFrame to store motion between frames
        cols = []
        for i in range(self.feature_params['maxCorners']):
            cols.append('p' + str(i) + '_x')
            cols.append('p' + str(i) + '_y')
        motion = pd.DataFrame(columns=cols)
        color = np.random.randint(0,255,(100,3))
        old_frame = self.video_buffer[start]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_gray, **self.feature_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        first_frame = True
        
        for idx in range(start, end) :
            frame = self.video_buffer[idx]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
        
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # save changes in 
            d_motion = {}
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                d_motion['p'+str(i)+'_x']=(a-c)*self.pix_width
                d_motion['p'+str(i)+'_y']=(b-d)*self.pix_heigth
                if first_frame:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    mask = cv2.putText(mask, str(i), (a,b), font, 1, color[i].tolist(), 1, lineType = cv2.LINE_AA)
                if not first_frame:
                    mask[:100, :400, :] = 0
                dx2dt = round(((a-c)*self.pix_width)*self.fps, 5)
                speed_x = "dx/dt = {0} m/s".format(dx2dt)
                dy2dt = round(((b-d)*self.pix_width)*self.fps, 5)
                speed_y = "dy/dt = {0} m/s".format(dy2dt)
                mask = cv2.putText(mask, speed_x, (5,45), font, 1, (180, 0, 80), 1, lineType = cv2.LINE_AA)
                mask = cv2.putText(mask, speed_y, (5,80), font, 1, (180, 0, 00), 1, lineType = cv2.LINE_AA)
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            motion = motion.append(d_motion, ignore_index=True)
            img = cv2.add(frame,mask)
            self.track_video.append(img)
            cv2.imshow('frame',img)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
            if k == ord("f"):
                idx+=1
        
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            first_frame=False
        cv2.destroyAllWindows()
        filename = self.video_file[:-4]+"_track_frames_{0}_to_{1}.csv".format(start,end)
        Tk().withdraw()
        save_filename = asksaveasfilename(initialfile=filename,
                                          defaultextension=".csv", 
                                          filetypes=[("coma seperate, salues", "csv"), 
                                                     ("all files", ".*")],
                                          title="Save motion records")
        if save_filename =="":
            return
        motion.to_csv(save_filename)
        return
    
    def save_tracking_video(self, fps=None):
        """
        Opens a tkinter save file dialog to get save file name
        and save a video under the given name
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        
        video=self.track_video
        filename = self.video_file[:-4]+"_track_.mp4"
        Tk().withdraw()
        save_file = asksaveasfilename(initialfile=filename,
                                      defaultextension=".mp4",
                                      filetypes=[("video file", "mp4"), 
                                                 ("all files", ".*")],
                                      title="Save motion video")
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                              frameSize=(self.width, self.height),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file
    
    def trim_video(self, trim):
        """
        Trim a cideo to between set frames. The frames are set by click "s" for start frame 
        and "e" for end frame. A track bar and "f" click - for forward one frame and
        "b" click for back one frame can be used to get to the start and end frames
        ---------------------------------------------------------
        Params:
            self
            trim (bool) - if True - trim the ogject video buffer
                          if False - keep the object video buffer 
                          and return start ans end frames
        return:
            start, end: (int) start and end frames numbers 
        """

        def _move(frame_n):
            self.frame_num=frame_n
            
        win_width, win_heigth = self.video_buffer[0].shape[:2]
        win_width, win_heigth = int(win_width/2), int(win_heigth/2)
        win_name = "Video trim. \
                    Click 's' for start frame, \
                    'e' for end frame, \
                    'f' for forward, \
                    'b' for backward and Esc to save and close"
        track_name = "Frame #:"
        start = 0
        end = self.length
        cv2.namedWindow(win_name)
        cv2.resizeWindow(win_name, win_width, win_heigth)
        cv2.createTrackbar(track_name, win_name,0,end, _move)
        self.frame_num = 0
        cv2.setTrackbarPos(track_name, win_name, self.frame_num)
        
        while True:
            frame = self.video_buffer[self.frame_num-1].copy()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(win_name, frame)
            k = cv2.waitKey(30) & 0xFF
            if k==ord("s"):
                start = self.frame_num
            if k==ord("e"):
                end = self.frame_num
            if k==ord("f"):
                self.frame_num = min(self.frame_num+1, self.length)
            if k==ord("b"):
                self.frame_num = max(0, self.frame_num-1)
            if k==27:
                break
            cv2.setTrackbarPos(track_name, win_name, self.frame_num)
        if trim:
            self.video_buffer = self.video_buffer[start:end,...]
            self.length = self.video_buffer.shape[0]
        cv2.destroyAllWindows()
        return start, end
        
        
    def play_video(self):
        def _move(frame_n):
            self.frame_num=frame_n

        win_width, win_heigth = int(self.width/2), int(self.height/2)
        win_name = "Play video: {0}". format(self.video_file)
        track_name = "Frame #:"
        cv2.namedWindow(win_name)
        cv2.resizeWindow(win_name, win_width, win_heigth)
        cv2.createTrackbar(track_name, win_name, 0, self.length, _move)
        self.frame_num = 0
        cv2.setTrackbarPos(track_name, win_name, self.frame_num)
        while True:
            f = self.video_buffer[self.frame_num]
            f = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(win_name, f)
            k=cv2.waitKey(int(1000/self.fps))&0xFF
            if k==27:
                break
            self.frame_num += 1
            if self.frame_num >= self.length:
                break
            cv2.setTrackbarPos(track_name, win_name, self.frame_num)
        cv2.destroyAllWindows()     

class Gui:
    def __init__(self):
        
        self.video_loaded = False
        self.window = Tk()
        self.window.wm_title('Video tracker')

        self.b1 = Button(self.window, text='Load Video', command=self.load_video, width=25)
        self.b1.grid(row=1, column=1)

        self.b2 = Button(self.window, text='Set area to track', command=self.get_roi, width=25)
        self.b2.grid(row=2, column=1)

        self.b3 = Button(self.window, text='Set pixel size (cm)', command=self.set_pxl_size, width=25)
        self.b3.grid(row=3, column=1)

        self.b4 = Button(self.window, text='Track motion', command=self.video_track, width=25)
        self.b4.grid(row=4, column=1)
        
        self.b5 = Button(self.window, text='Trim video', command=self.trim_video, width=25)
        self.b5.grid(row=5, column=1)

        self.b6 = Button(self.window, text='Save video', command=self.save_tracked, width=25)
        self.b6.grid(row=6, column=1)

        self.b7 = Button(self.window, text='Play video', command=self.play_video, width=25)
        self.b7.grid(row=7, column=1)
        
        self.b8 = Button(self.window, text='Exit', command=self.quite, width=25)
        self.b8.grid(row=8, column=1)

        self.window.mainloop()
        
    def load_video(self):
        
        self.video = Video_utill()
        self.video_loaded = True
        
    def get_roi(self):
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            self.video.set_roi()
        
    def set_pxl_size(self):
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            self.video.set_pxl_size()
        
    def video_track(self):  
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            self.video.track()
            
    def save_tracked(self):
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            self.video.save_tracking_video(fps=30)

    def trim_video(self):
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            _, _ =self.video.trim_video(trim=True)

    def play_video(self):
        if not self.video_loaded:
            messagebox.showerror('No video selected error', 'Please load video first')
        else:
            self.video.play_video()   

    def quite(self):
        self.window.destroy()


if __name__=='__main__':
    #g = Gui()
    VIDEO_FILE=os.path.join(os.path.dirname(os.getcwd()),"data","00001_Trim.mp4")
    v=Video_utill(VIDEO_FILE)
