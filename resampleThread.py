import sys
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QTimer,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont, QMovie, QCursor,QPainter, QPen,QPixmap, QTransform, QIcon)

import psutil
import pyqtgraph as pg

from adaptive_balance import adaptive_balance
from balancefringe import fringes
from processParams import processParams
from resample import linear_k
from skimage import exposure
from processingThread import ProcessingThread
from TSAThread import TSAThread
from dispersion import *
import matplotlib.animation as animation
import registration

from scipy.signal.windows import tukey

import pynvml
import time,os
import sys,cv2
import numpy as np


from PIL import Image,ImageEnhance,ImageOps,ImageQt
from octFuncs import *
from random import randint
from sidebar_ui import Ui_MainWindow



class ResampleThread(QThread):
    finished = Signal()
    progress = Signal(int)
    resample_ready = Signal(np.ndarray,list,list)

    def __init__(self, processParameters,frame_3D,coord_one,coord_two,width):
        self.processParameters = processParameters
        self.frame_3D = frame_3D
        self.width = width
        self.coord_one = coord_one
        self.coord_two = coord_two
        super().__init__()

    def map_to_real(self,point):
        return np.array((int(self.processParameters.res_fast*point[0]/1024),int(self.processParameters.res_slow*point[1]/1024)))
    
    def get_angle(self, center, p2, zero_rad):
        vector1 = zero_rad
        vector2 = (p2[0] - center[0], p2[1] - center[1])
        
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        
        # Calculate the magnitudes of the two vectors
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        
        # Calculate the angle between the two lines using the dot product formula
        angle = math.acos(dot_product / (magnitude1 * magnitude2))
        

        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if cross_product > 0:
            angle = 2*math.pi - angle
        return round(angle,3)

    def aline_reg(self, aline_stack, aline, averaged=True):
        ret = cp.zeros((np.shape(aline_stack)))
        if averaged: 
            aline1 = cp.fft.fft(aline)
            for i in range(np.shape(aline_stack)[1]):
                aline2 = cp.fft.fft(aline_stack[:,i])
                crs_cor = cp.abs(cp.fft.ifft(aline1*cp.conj(aline2)))
                shift = cp.argmax(crs_cor)
                ret[:,i] = cp.roll(aline_stack[:,i],shift)
            return cp.squeeze(cp.mean(ret,axis=1))
        else:
            aline1 = cp.fft.fft(cp.squeeze(cp.mean(aline,axis=1)))
            shift = cp.zeros(np.shape(aline_stack)[1])
            for i in range(1,np.shape(aline_stack)[1]):
                aline2 = cp.fft.fft(aline_stack[:,i])
                crs_cor = cp.abs(cp.fft.ifft(aline1*cp.conj(aline2)))
                
                shift[i] = cp.argmax(crs_cor)
                if shift[i] >= 512:
                    print("overflow:",i)
                    shift[i] = shift[i] -1024
            
            #mean = 0
            #for i in range(cp.shape(shift)[0]):
            #    if cp.abs(shift[i]) > 200:
            #        shift[i] = int(mean)
            #out_shift = cupyx.scipy.signal.medfilt(shift,15)
            
            for i in range(cp.shape(shift)[0]):
                ret[:,i] = cp.roll(aline_stack[:,i],int(shift[i]))
            
            
            return ret

    def run(self):
        import math
        from collections import OrderedDict
        import matplotlib.patches as patches
        processParameters = self.processParameters
        frame_3D = self.frame_3D
        width = self.width
        x_mesh = np.arange(min(self.coord_one[0],self.coord_two[0]),max(self.coord_one[0],self.coord_two[0])+1)
        y_mesh = np.arange(min(self.coord_one[1],self.coord_two[1]),max(self.coord_one[1],self.coord_two[1])+1)
        
        if self.coord_one[0] < self.coord_two[0]:
            slope = float((self.coord_one[1] - self.coord_two[1])/(self.coord_one[0] - self.coord_two[0]))
        else:
            slope = float((self.coord_two[1] - self.coord_one[1])/(self.coord_two[0] - self.coord_one[0]))
        intercept = self.coord_one[1] - slope * self.coord_one[0]
        print(intercept)
        circ_points = []
        print(self.coord_one)
        print(self.coord_two)
        if np.shape(x_mesh)[0] > np.shape(y_mesh)[0]:
            frame = np.zeros((1024,np.shape(x_mesh)[0]))
            for i in range(np.shape(x_mesh)[0]):
                frame[:,i] = frame_3D[:,int(x_mesh[i]),int(slope* x_mesh[i] + intercept)]
        else:
            frame = np.zeros((1024,np.shape(y_mesh)[0]))
            for i in range(np.shape(y_mesh)[0]):
                frame[:,i] = frame_3D[:,int((y_mesh[i]-intercept)/slope),int(y_mesh[i])]
            print([int((y_mesh[0]-intercept)/slope),y_mesh[0]])
            print([int((y_mesh[-1]-intercept)/slope),y_mesh[-1]])
        #print("zero rad is:",zero_rad)
        #print(radiant_samples)
        #sobelxy = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        if self.coord_one[0] > self.coord_two[0]:
            frame = np.fliplr(frame)
        self.resample_ready.emit(frame,self.coord_one,self.coord_two)
        print("finished")
