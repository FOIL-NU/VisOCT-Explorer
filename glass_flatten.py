from scipy.interpolate import UnivariateSpline
import sys
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QSize,QRect,QCoreApplication)
import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

class glass_flatten(QThread):
    progress = Signal(int)
    flatten_ready = Signal(np.ndarray)

    def __init__(self,frame_3D_resh):
        self.frame_3D_oct = frame_3D_resh
        super().__init__()

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


    def run(self):
        frame_3D_oct = self.frame_3D_oct
        enface_slow = np.squeeze(np.max(frame_3D_oct,axis=1))
        enface_slow = enface_slow/np.mean(enface_slow,axis=0)
        enface_slow = enface_slow/np.max(enface_slow)

        

        enface_slow = (((np.clip((enface_slow - 0) / (1 - 0), 0, 1)) * 255)).astype(np.uint8)

        filtered_img = cv2.medianBlur(enface_slow, ksize=3)

        _, binary_image = cv2.threshold(filtered_img,30, 255, cv2.THRESH_BINARY)
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
            x_array = []
            y_array = []
            for i in topmost_pix:
                #print(i)
                x_array.append(i[0])
                y_array.append(i[1])
            degree = 1

            # Fit the data to a polynomial of the specified degree
            coefficients = np.polyfit(x_array, y_array, degree)

            # Create a polynomial function based on the coefficients
            poly = np.poly1d(coefficients)

            # Generate points for the polynomial curve
            x_fit = np.linspace(0, np.shape(enface_slow)[1]-1, np.shape(enface_slow)[1])  # Generating x-values for the fit
            y_fit = poly(x_fit)  # Calculating y-values for the fit

        y_shift = np.zeros(np.shape(enface_slow)[1])
        for i in range(np.shape(enface_slow)[1]):
            y_shift[i] = int(y_fit[i]-y_fit[int(np.shape(enface_slow)[1]/2)])

        frame_3D_oct_new = frame_3D_oct.copy()
        for i in range(np.shape(enface_slow)[1]):
            frame_3D_oct_new[:,:,i] = np.roll(frame_3D_oct_new[:,:,i],-int(y_shift[i]),axis=0)
            
        enface_slow = np.squeeze(np.max(frame_3D_oct_new,axis=1))
        enface_slow = enface_slow/np.mean(enface_slow,axis=0)
        enface_slow = enface_slow/np.max(enface_slow)

        enface_fast = np.squeeze(np.max(frame_3D_oct_new,axis=2))
        enface_fast = enface_fast/np.mean(enface_fast,axis=0)
        enface_fast = enface_fast/np.max(enface_fast)
        enface_fast = (((np.clip((enface_fast - 0) / (1 - 0), 0, 1)) * 255)).astype(np.uint8)

        filtered_img = cv2.medianBlur(enface_fast, ksize=3)

        _, binary_image = cv2.threshold(filtered_img,30, 255, cv2.THRESH_BINARY)
        plt.imshow(binary_image,cmap='gray')


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
            x_array = []
            y_array = []
            for i in topmost_pix:
                #print(i)
                x_array.append(i[0])
                y_array.append(i[1])
            degree = 1

            # Fit the data to a polynomial of the specified degree
            coefficients = np.polyfit(x_array, y_array, degree)

            # Create a polynomial function based on the coefficients
            poly = np.poly1d(coefficients)

            # Generate points for the polynomial curve
            x_fit = np.linspace(0, np.shape(enface_fast)[1]-1, np.shape(enface_fast)[1])  # Generating x-values for the fit
            y_fit = poly(x_fit)  # Calculating y-values for the fit
        y_shift = np.zeros(np.shape(enface_fast)[1])
        for i in range(np.shape(enface_fast)[1]):
            y_shift[i] = int(y_fit[i]-y_fit[int(np.shape(enface_fast)[1]/2)])

        for i in range(np.shape(frame_3D_oct_new)[2]):
            for j in range(np.shape(frame_3D_oct_new)[1]):
                frame_3D_oct_new[:,j,i] = np.roll(frame_3D_oct_new[:,j,i],-int(y_shift[j]))


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
        self.flatten_ready.emit(frame_3D_oct_new)

        return