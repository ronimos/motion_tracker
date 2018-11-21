# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:10:16 2018

@author: Ron Simenhois
"""

import numpy as np
import cv2
from tkinter import Tk, simpledialog, Button, messagebox
from tkinter.filedialog import askopenfilename
import pandas as pd

'''
Video_utill Class: measure motion of items throughout a video
'''

class Video_utill:
    def __init__(self, video_file=None):
        '''
        Load a video file into a numpty array and save the video parameters
        params:
            video_file - (str) name of the video file to load. if there is no video name
                               an ask for file window is opened to chuse a file.
        return:
            None
        '''       
        if video_file is None:
            Tk().withdraw()
            self.video_file = askopenfilename()
            self.cap = cv2.VideoCapture(self.video_file)
        else:
            self.video_file = video_file
            # params for ShiTomasi corner detection
        self.feature_params = dict(smaxCorners = 10,
                                   qualityLevel = 0.3,
                                   minDistance = 5,
                                   blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.pxl_size = -1
        self.load_video()
        self.mask = self.video_buffer[0].copy()
        self.pix_heigth = None
        
    def load_video(self):
        '''
        This function is called from __init__() and load a video file into and numoy array
        '''
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_buffer = np.zeros((self.length, self.height, self.width, 3), dtype='uint8')
        for i in range(self.length):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.video_buffer[i]=frame.copy()
        self.cap.release()
        return

    def on_mouse(self,event, x, y, flag, param):
        '''
        This function is called on mouse event when a set RIO or pixel size 
        calculation windows are open. it draw a rectangle on an image and save 
        its location and size
        Params:
            self
            event - (int) event (mouse move, r click, l click...) code
            x, y - (int, int) - mouse's x,y location
            flags
            params
        '''
        if self.drawing:
            self.img_cp = self.img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.rectangle(self.img_cp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                cv2.rectangle(self.img_cp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                self.ul_cr = (max(min(x, self.ix), 0), max(min(y, self.iy), 0))
                self.lr_cr = (min(max(x, self.ix), self.img_cp.shape[1]), min(max(y, self.iy), self.img_cp.shape[0]))
        return

        
    
    def set_roi(self):
        '''
        Open a window with the first video's frame and call on_mouse to draw a rectangle
        and save it as ROI for area to track
        '''
        self.mask = np.zeros_like(self.video_buffer[0])
        zoom = 1
        self.drawing = False
        window_name = 'Draw a rectangle with around the area you want to track, Click "-" to resize, Esc to exit'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)
        self.img = self.video_buffer[0].copy()
        self.img_cp = self.img.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])
        while True:
            cv2.imshow(window_name, self.img_cp)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            if k == ord('-'):
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_cp = self.img.copy()
                self.ul_cr = (0, 0)
                self.lr_cr = (self.img.shape[1], self.img.shape[0])
                zoom *= 2
        cv2.destroyAllWindows()
        self.mask[self.ul_cr[1]*zoom:self.lr_cr[1]*zoom, self.ul_cr[0]*zoom:self.lr_cr[0]*zoom, :] = self.video_buffer[0,self.ul_cr[1]*zoom:self.lr_cr[1]*zoom, self.ul_cr[0]*zoom:self.lr_cr[0]*zoom, :]
        return
    
    
    def set_pxl_size(self):
        '''
        Open a window with the first video's frame and call on_mouse to draw a rectangle,
        Open an input window to enter the rectangle size in cm and calculates pixel zise
        '''        
        zoom = 1
        self.drawing = False
        window_name = 'Draw a rectangle with a known height (in cm), Click "-" to resize, Esc to exit'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)
        self.img = self.video_buffer[0].copy()
        self.img_cp = self.img.copy()
        self.ul_cr = (0, 0)
        self.lr_cr = (self.img.shape[1], self.img.shape[0])
        while True:
            cv2.imshow(window_name, self.img_cp)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            if k == ord('-'):
                self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.img_cp = self.img.copy()
                self.ul_cr = (0, 0)
                self.lr_cr = (self.img.shape[1], self.img.shape[0])
                zoom *= 2
        cv2.destroyAllWindows()
        heigth = np.abs(self.ul_cr[0] - self.lr_cr[0]) * zoom
        rect_height = simpledialog.askfloat('Input', 'Insert rectangle heigth in cm')
        self.pix_heigth = rect_height/heigth
        return
            
        
    def track(self):
        '''
        Uses Optical Flow to track movment of features inside the ROI and save
        the motion between frames (on both x,y axis) in cm if pixel zise is set or in 
        pixels if not
        '''
        # Create DataFrame to store motion between frames
        cols = []
        if self.height is None:
            p_size = 1
            unit = 'pxl'
        else: #self.height is set
            p_size = self.height
            unit = 'cm'
        for i in range(self.feature_params['maxCorners']):
            cols.append('p' + str(i) + '_x ({})'.format(unit))
            cols.append('p' + str(i) + '_y ({})'.format(unit))
        motion = pd.DataFrame(columns=cols)
        color = np.random.randint(0,255,(100,3))
        self.cap = cv2.VideoCapture(self.video_file)
        # Take first frame and find corners in it
        ret, old_frame = self.cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_gray, **self.feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(True):
            ret,frame = self.cap.read()
            if not ret:
                break
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
                d_motion['p'+str(i)+'_x']=(a-c)*p_size
                d_motion['p'+str(i)+'_y']=(b-d)*p_size
                
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            motion = motion.append(d_motion, ignore_index=True)
            img = cv2.add(frame,mask)
        
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        cv2.destroyAllWindows()
        file_name = simpledialog.askstring('Input', 'Insert file name to save video motion') + '.csv'
        motion.to_csv(file_name)
        return
    
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


if __name__=='__main__':
    g = Gui()
