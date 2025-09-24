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

def imcontrast_adjust(img):

    imgmin = np.nanmean(np.min(img,axis=0))
    imgmax = np.nanmean(np.max(img,axis=0))

    if np.isnan(imgmin) or np.isnan(imgmax):
        img = img
    else:
        img = ((np.clip((img - imgmin) / (imgmax+50-imgmin), 0, 1))*255).astype('uint8')
        
    return img

class FibergramThread(QThread):
    progress = Signal(int)
    fibergram_ready = Signal(np.ndarray,np.ndarray)

    def __init__(self, frame_3D_resh_log,frame_3D_resh,low_Val,high_Val,threshold_Val):
        self.frame_3D_resh_log = frame_3D_resh_log
        self.frame_3D_resh = frame_3D_resh
        self.low_Val = low_Val
        self.high_Val = high_Val
        self.threshold_Val = 75
        super().__init__()



    def run(self):
        frame_3D_resh_log = self.frame_3D_resh_log
        frame_3D_resh = self.frame_3D_resh
        low_Val = 68
        high_Val = 76
        threshold_Val = self.threshold_Val
        fibergram = np.zeros((1024,np.shape(frame_3D_resh_log)[1],np.shape(frame_3D_resh_log)[2]-20))
        fibergram[:] = np.nan
        fitted_top_boundary = np.zeros((np.shape(frame_3D_resh_log)[1],np.shape(frame_3D_resh_log)[2]))

        #Prepare the flattening matrix for all aline in all frames
        for frame_number in range(8,np.shape(frame_3D_resh_log)[2]-12):
            self.progress.emit(int(np.ceil((frame_number-8)/(np.shape(frame_3D_resh_log)[2]-20)*100)))
        #for frame_number in range(77,78):
            to_GPU = registration.shift_correction_bulk(frame_3D_resh_log[:,:,frame_number:frame_number+5],5)
            image = (((np.clip((np.mean(cp.asnumpy(to_GPU),axis=2) - low_Val) / (high_Val - low_Val), 0, 1)) * 255)).astype(np.uint8)
            filtered_img = cv2.medianBlur(image, ksize=3)
            
            #plt.imsave('D:\OCT_annotation_patients\oct_420.png',((np.clip((frame - octa_lowVal_new) / (octa_highVal - octa_lowVal_new), 0, 1)) * 255).astype(np.uint8),cmap='gray')
            #print(enface[0,:])
            _, binary_image = cv2.threshold(filtered_img, threshold_Val, 255, cv2.THRESH_BINARY)
            
            #plt.imshow(binary_image,cmap='gray')
            #plt.show()

            topmost_pixels = self.find_topmost_pixels(binary_image)

            if topmost_pixels:
                topmost_pix = []
                # Group the top-most pixels by unique X values
                unique_x_values = np.unique([pixel[0] for pixel in topmost_pixels])

                for x_value in unique_x_values:
                    # Find the top-most pixel for the current X value
                    topmost_pixel = min((x, y) for x, y in topmost_pixels if x == x_value)
                    #print(f"X = {x_value}, Top-most pixel coordinates: {topmost_pixel}")
                    topmost_pix.append(topmost_pixel)
            else:
                print("No connected areas found.")
            temp_img = np.zeros((1024,1024))
            x_array = []
            y_array = []
            for i in topmost_pix:
                #print(i)
                x_array.append(i[0])
                y_array.append(i[1])
                temp_img[i[1],i[0]] = 252
                

            # Sample data (replace this with your own dataset)


            # Degree of the polynomial (e.g., 2 for a quadratic polynomial)
            degree = 15

            # Fit the data to a polynomial of the specified degree
            coefficients = np.polyfit(x_array, y_array, degree)

            # Create a polynomial function based on the coefficients
            poly = np.poly1d(coefficients)

            # Generate points for the polynomial curve
            x_fit = np.linspace(0, np.shape(image)[1]-1, np.shape(image)[1])  # Generating x-values for the fit
            y_fit = poly(x_fit)  # Calculating y-values for the fit

            new_topmost_pix = []
            for i in range(0,np.shape(topmost_pix)[0]):
                #print(topmost_pix[i])
                if topmost_pix[i][1] < poly(topmost_pix[i][0]) :
                    new_topmost_pix.append(topmost_pix[i])


            x_array = []
            y_array = []
            for i in new_topmost_pix:
                #print(i)
                x_array.append(i[0])
                y_array.append(i[1])
                
            spl = UnivariateSpline(x_array, y_array)
            #spl.set_smoothing_factor(0.5)
            #poly = np.poly1d(new_coefficients)
            x_fit = np.linspace(0, np.shape(image)[1]-1, np.shape(image)[1])  # Generating x-values for the fit
            y_fit = spl(x_fit)  # Calculating y-values for the fit

            #print("fibergram shape",np.shape(fibergram))
            #print("frame_3D_resh shape",np.shape(frame_3D_resh))
            
            for x_coord in range(np.shape(image)[1]):
                #print("y fit:",y_fit[x_coord])
                fitted_top_boundary[x_coord,frame_number] = min(max(y_fit[x_coord],0),1023)
                fibergram_y_fit_value = y_fit[x_coord]
                #print("fibergram_y_fit",fibergram_y_fit_value)
                fibergram[int(fibergram_y_fit_value):int(fibergram_y_fit_value)+10,x_coord,frame_number-8] = frame_3D_resh[int(y_fit[x_coord]):int(y_fit[x_coord])+10,x_coord,frame_number+2]

        #commented out the fitting along the other axis
        '''
        for x in range(np.shape(image)[1]):
            fast_axis_y = fitted_top_boundary[x,:]
            fast_axis_x = np.linspace(0, np.shape(frame_3D_resh_log)[2]-1, np.shape(frame_3D_resh_log)[2])    
            spl = UnivariateSpline(fast_axis_x, fast_axis_y)
            y_fit = spl(fast_axis_x)
            fitted_top_boundary[x,:] = y_fit
            
        for frame_number in range(0,np.shape(frame_3D_resh_log)[2]):
        #for frame_number in range(77,78):
            #image = (((np.clip((frame_3D_resh_log[:,:,frame_number] - octa_lowVal_new) / (octa_highVal_new - octa_lowVal_new), 0, 1)) * 255)).astype(np.uint8)
            image = frame_3D_resh[:,:,frame_number+15]
            plt.imshow(image,cmap='gray')
            x_fit = np.linspace(0, 494, 495)
            plt.plot(x_fit+17, fitted_top_boundary[:,frame_number], label='Fitted Polynomial', color='red')
            #plt.legend()
            plt.title('Polynomial Fit')
            print(frame_number)
            plt.show()
            for x_coord in range(495):
                fibergram[int(fitted_top_boundary[x_coord,frame_number]):int(fitted_top_boundary[x_coord,frame_number])+10,x_coord,frame_number] = image[int(fitted_top_boundary[x_coord,frame_number]):int(fitted_top_boundary[x_coord,frame_number])+10,x_coord+17]
        '''
        #endTime = time.time()
        fibergram_enface = np.squeeze(np.nanmean(fibergram,axis=0))
        self.fibergram_ready.emit(fibergram_enface,fitted_top_boundary)

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
