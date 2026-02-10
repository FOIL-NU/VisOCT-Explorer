import sys
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QTimer,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont, QMovie, QCursor,QPainter, QPen,QPixmap, QTransform, QIcon)

import psutil
import pyqtgraph as pg

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
    resample_ready = Signal(np.ndarray,list,list,np.ndarray)

    def __init__(self, processParameters,frame_3D,radius_given,center,width):
        self.processParameters = processParameters
        self.frame_3D = frame_3D
        self.radius_given = radius_given
        self.width = width
        self.circ_center = center
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
        return round(angle,4)

    def update_resample_content(self, processParameters, frame_3D, radius_given,center, width):
        self.processParameters = processParameters
        self.frame_3D = frame_3D
        self.radius_given = radius_given
        self.width = width
        self.circ_center = center

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
            #aline1 = cp.fft.fft(cp.squeeze(cp.mean(aline,axis=1)))
            aline1 = cp.fft.fft(aline[:,0])
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
        x_pixel_width = processParameters.xrng/processParameters.res_fast
        radius_given = int(self.radius_given/2)
        circ_center = np.array((int(self.circ_center[0]/2),int(self.circ_center[1]/2)))
        radius_real = radius_given * x_pixel_width
        width = self.width
        radius_width = []
        for i in range(width+1):
            radius_width.append(radius_given-width/2+i)
        if circ_center[0] - radius_given < 0 and circ_center[1] - radius_given < 0 and circ_center[0] + radius_given > 1024 and circ_center[1] + radius_given > 1024:
            print("set better radius")

        if (circ_center[0] - radius_given >= 0 and circ_center[0] + radius_given < 1024) and (circ_center[1] - radius_given >= 0 and circ_center[1] - radius_given < 1024):
            zero_x = -1
            zero_y = 1
            zero_rad = (-10,0)
            rad_min = 0
            rad_max = 6.2832
            start_point = (circ_center[0]*2-radius_given*2,circ_center[1]*2)
            end_point = (circ_center[0]*2-radius_given*2,circ_center[1]*2)
        else:
            if (circ_center[0] - radius_given < 0):
                zero_x = -1
                zero_y = 1
                zero_rad = (-10,0)
            elif (circ_center[0] + radius_given >= 1024):
                zero_x = 1
                zero_y = -1
                zero_rad = (10,0)
            elif (circ_center[1] + radius_given >= 1024):
                zero_x = -1
                zero_y = -1
                zero_rad = (0,10)
            elif (circ_center[1] - radius_given < 0):
                zero_x = 1
                zero_y = 1
                zero_rad = (0,-10)

            if circ_center[0] - radius_given < 0:
                if circ_center[1] - radius_given < 0: #bottom left corner
                    start_point = (0,circ_center[1] + np.sqrt(radius_given**2 - circ_center[0]**2))
                    end_point = (circ_center[0] + np.sqrt(radius_given**2 - circ_center[1]**2),0)
                elif circ_center[1] + radius_given > 1023: #top left corner
                    start_point = (circ_center[0] + np.sqrt(radius_given**2 - (1023-circ_center[1])**2),1023)
                    end_point = (0,circ_center[1] - np.sqrt(radius_given**2 - circ_center[0]**2))
                else: #to the left edge
                    start_point = (0,circ_center[1] + np.sqrt(radius_given**2 - circ_center[0]**2))
                    end_point = (0,circ_center[1] - np.sqrt(radius_given**2 - circ_center[0]**2))
            elif circ_center[0] + radius_given > 1023:
                if circ_center[1] - radius_given < 0: #bottom right corner
                    start_point = (circ_center[0] - np.sqrt(radius_given**2 - circ_center[0]**2),0)
                    end_point = (1023,circ_center[1] + np.sqrt(radius_given**2 - (1023-circ_center[0])**2))
                elif circ_center[1] + radius_given > 1023: #top right corner
                    start_point = (1023,circ_center[1] - np.sqrt(radius_given**2 - (1023-circ_center[0])**2))
                    end_point = (circ_center[0] - np.sqrt(radius_given**2 - (1023-circ_center[1])**2),1023)
                else: #to the right edge
                    start_point = (1023,circ_center[1] - np.sqrt(radius_given**2 - (1023-circ_center[0])**2))
                    end_point = (1023,circ_center[1] + np.sqrt(radius_given**2 - (1023-circ_center[0])**2))
            elif circ_center[1] - radius_given < 0: #to the bottom edge
                start_point = (circ_center[0] - np.sqrt(radius_given**2 - circ_center[1]**2),0)
                end_point = (circ_center[0] + np.sqrt(radius_given**2 - circ_center[1]**2),0)
            elif circ_center[1] + radius_given > 1023: #to the top edge
                start_point = (circ_center[0] + np.sqrt(radius_given**2 - (1023-circ_center[1])**2),1023)
                end_point = (circ_center[0] - np.sqrt(radius_given**2 - (1023-circ_center[1])**2),1023)

            rad = np.array((self.get_angle(circ_center,start_point,zero_rad), self.get_angle(circ_center,end_point,zero_rad)))
            '''if (circ_center[1] - radius_given < 0 and circ_center[1] + radius_given >= 1024):
                rad = np.array((self.get_angle(circ_center,np.array((circ_center[0] + np.sqrt(radius_width[-1]**2 - circ_center[1]**2),0)),zero_rad), self.get_angle(circ_center,np.array((circ_center[0] + np.sqrt(radius_width[-1]**2 - (1023-circ_center[1])**2),1023)),zero_rad)))

            elif circ_center[1] - radius_given < 0:
                #rad = np.array((self.get_angle(circ_center,np.array((circ_center[0] + np.sqrt(radius_width[-1]**2 - circ_center[1]**2),0)),zero_rad), self.get_angle(circ_center,np.array((circ_center[1] + np.sqrt(radius_width[-1]**2 - circ_center[0]**2),0)),zero_rad)))
                rad = np.array((self.get_angle(circ_center,np.array((circ_center[0] + np.sqrt(radius_width[-1]**2 - circ_center[1]**2),0)),zero_rad), 0))
            elif circ_center[1] - radius_given >= 0 and circ_center[1] + radius_given >= 1024:
                rad = np.array((self.get_angle(circ_center,np.array((0,circ_center[1] - np.sqrt(radius_width[-1]**2 - circ_center[0]**2))),zero_rad), self.get_angle(circ_center,np.array((circ_center[0] + np.sqrt(radius_width[-1]**2 - (1023-circ_center[1])**2),1023)),zero_rad)))

            else:
                rad = np.array((self.get_angle(circ_center,np.array((0,circ_center[1] - np.sqrt(radius_width[-1]**2 - circ_center[0]**2))),zero_rad), self.get_angle(circ_center,np.array((0,circ_center[1] + np.sqrt(radius_width[-1]**2 - circ_center[0]**2))),zero_rad)))'''

            rad_min = np.min(rad)
            rad_max = np.max(rad)
        num_sample = int((rad_max-rad_min)*radius_real/x_pixel_width)
        radiant_samples = np.linspace(rad_min,rad_max,num=num_sample)
        #print("zero rad is:",zero_rad)
        #print(radiant_samples)

        circ_resampled = {}
        count = 0

        #frame = np.zeros((1024,np.shape(radiant_samples)[0]))
        # for theta in radiant_samples:
        #     self.progress.emit(int(100*count/np.shape(radiant_samples)[0]))
        #     #temp = cp.zeros((1024,np.shape(radius_width)[0]))
        #     if zero_rad[0] != 0:
        #         x = round(zero_x * radius_given * math.cos(theta) + circ_center[0])
        #         y = round(zero_y * radius_given * math.sin(theta) + circ_center[1])
        #     else:
        #         x = round(zero_x * radius_given * math.sin(theta) + circ_center[0])
        #         y = round(zero_y * radius_given * math.cos(theta) + circ_center[1])
        #     cur_lines = cp.zeros((1024,9))
        #     cur_lines[:,0] = cp.asarray(frame_3D[:,x,y])
        #     cur_lines[:,1] = cp.asarray(frame_3D[:,x,y-1])
        #     cur_lines[:,2] = cp.asarray(frame_3D[:,x,y+1])
        #     cur_lines[:,3] = cp.asarray(frame_3D[:,x-1,y-1])
        #     cur_lines[:,4] = cp.asarray(frame_3D[:,x-1,y])
        #     cur_lines[:,5] = cp.asarray(frame_3D[:,x-1,y+1])
        #     cur_lines[:,6] = cp.asarray(frame_3D[:,x+1,y])
        #     cur_lines[:,7] = cp.asarray(frame_3D[:,x+1,y-1])
        #     cur_lines[:,8] = cp.asarray(frame_3D[:,x+1,y+1])
        #     result = self.aline_reg(cur_lines,cur_lines[:,0],averaged=True)
        #     print("11:",np.shape(result))
        #     frame[:,count] = result.get()
        #     print("111")
        #     count += 1
            

        delta = 8
        for theta in radiant_samples:
            
            temp = cp.zeros((1024,np.shape(radius_width)[0]))
            
            #width_count = 0
            for rad in radius_width:
                if zero_rad[0] != 0:
                    x = round(zero_x * rad * math.cos(theta) + circ_center[0])
                    y = round(zero_y * rad * math.sin(theta) + circ_center[1])
                else:
                    x = round(zero_x * rad * math.sin(theta) + circ_center[0])
                    y = round(zero_y * rad * math.cos(theta) + circ_center[1])
                    
                x = min(max(0,x),cp.shape(frame_3D)[1]-1)
                y = min(max(0,y),cp.shape(frame_3D)[2]-1)
                        
                angle = self.get_angle(circ_center,[x,y],zero_rad)
                if angle not in circ_resampled.keys():
                    circ_resampled[angle] = [x,y]
                    count += 1
                    
        frame = np.zeros((1024,count))
        circ_points = []
        count = 0
        for i in sorted(circ_resampled.keys()):
            print("angle is:",i)
            # need to think of it. why shape[1] - x - 1,shape[2] - y -1?
            circ_points.append(((int(circ_resampled[i][0])),int((circ_resampled[i][1]))))
            count += 1
            
        count = 0

        for coord_index in range(0,len(circ_points)-delta):
            self.progress.emit((int)(100*coord_index/(len(circ_points)-delta-1)))
            cur_lines = cp.zeros((1024,delta))
            for i in range(0,delta):
                #print("circ_points index x:",circ_points[coord_index+i][0],"y:",circ_points[coord_index+i][1])
                #print("angle is:",self.get_angle(circ_center,circ_points[coord_index+i],zero_rad))
                cur_lines[:,i] = cp.asarray(frame_3D[:,circ_points[coord_index+i][0],circ_points[coord_index+i][1]])
            result = self.aline_reg(cur_lines,cur_lines,averaged=False)
            print(count)
            frame[:,count] = cp.squeeze(cp.mean(result,axis=1)).get()
            count += 1

        print("circ center:",circ_center)
        print("radius real:",radius_given)
        #sobelxy = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        self.resample_ready.emit(frame,start_point,end_point,circ_center*2)
        print("finished")