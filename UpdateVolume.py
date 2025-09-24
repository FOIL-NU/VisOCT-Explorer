from scipy.interpolate import UnivariateSpline
import sys
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QSize,QRect,QCoreApplication)
from PySide6.QtGui import (QColor, QFont,QPixmap, QMovie)
from PySide6.QtWidgets import (QApplication,QSlider, QTableWidget, QTableWidgetItem, QDialog, QMessageBox, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,QPushButton, QFileDialog, QComboBox,QTextEdit)
from sidebar_ui import NumberOnlyTextEdit
from pyqtgraph import PlotWidget,GraphicsLayoutWidget,PlotItem,widgets,GraphicsLayout,GraphicsView
import registration
import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


class UpdateVolumeThread(QThread):
    progress = Signal(int)
    volume_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()

    def set_attrib(self,frame_3D_resh,processParameters,average_number):
        self.frame_3D_resh = frame_3D_resh
        self.processParameters = processParameters
        self.average_number = average_number

    def run(self):
        frame_3D_resh = self.frame_3D_resh
        processParameters = self.processParameters
        average_number = self.average_number
        
        chunk_num = 2
        per_chunk = int(processParameters.res_slow/chunk_num)
        
        self.progress.emit(int(10))
        if average_number <= int(cp.shape(frame_3D_resh)[2]/2)+4:
            temp = cp.zeros((cp.shape(frame_3D_resh)[0],cp.shape(frame_3D_resh)[1],int(cp.shape(frame_3D_resh)[2]/average_number)))
            for i in range(chunk_num):
                to_GPU = registration.shift_correction_bulk(frame_3D_resh[:,:,i*per_chunk:(i+1)*per_chunk],average_number)
                for frame_number in range(int(cp.shape(to_GPU)[2]/average_number)):
                    temp[:,:,i*(per_chunk/average_number)+frame_number] = cp.squeeze(cp.mean(to_GPU[:,:,(frame_number*average_number):(frame_number*average_number+average_number)],axis=2))
                    
                del to_GPU
                self.progress.emit(int(10+45*(i+1)))
                cp.get_default_memory_pool().free_all_blocks()
        else:
            temp = cp.zeros((cp.shape(frame_3D_resh)[0],cp.shape(frame_3D_resh)[1],1))
            to_GPU = registration.shift_correction_bulk(frame_3D_resh[:,:,:],int(cp.shape(frame_3D_resh)[2]))
            temp[:,:,0] = cp.squeeze(cp.mean(to_GPU[:,:,:],axis=2))
            del to_GPU
            self.progress.emit(int(100))
            cp.get_default_memory_pool().free_all_blocks()
        frame_3D_resh = cp.asnumpy(temp)
        self.volume_ready.emit(frame_3D_resh)
        return

        #fiber_enface = np.nanmean(fibergram,axis=0)



        #fiber_enface = imcontrast_adjust(fiber_enface)
        # fibergram =  cv2.normalize(fibergram, None, 0, 1, cv2.NORM_MINMAX)
        #fiber_enface = cv2.cvtColor(fiber_enface, cv2.COLOR_GRAY2BGR)
        #plt.imshow(fiber_enface, 'gray')
        #plt.colorbar()

    def find_topmost_pixels(self,binary_image):
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate areas for each contour
        areas = [cv2.contourArea(contour) for contour in contours]

        # Find the indices of the largest and second-largest areas
        indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)[:2]

        # Get the largest and second-largest contours
        largest_contour = contours[indices[0]]
        second_largest_contour = contours[indices[1]]
            

        if not contours:
            return None

        topmost_pixels = []
        largest_contour = max(contours, key=cv2.contourArea)

        for contour in largest_contour:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the top-most pixel coordinates
            topmost_pixel = (x, y)

            topmost_pixels.append(topmost_pixel)
            
        if cv2.contourArea(second_largest_contour) > 0.2*cv2.contourArea(largest_contour):
                
            for contour in second_largest_contour:
                    # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                    # Calculate the top-most pixel coordinates
                topmost_pixel = (x, y)

                topmost_pixels.append(topmost_pixel)

        # Display the image with the largest and second-largest contours

        return topmost_pixels
