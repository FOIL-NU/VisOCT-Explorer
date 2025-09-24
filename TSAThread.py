import sys

import psutil
import pyqtgraph as pg

from balancefringe import fringes
from processParams import processParams
from resample import linear_k
from skimage import exposure,morphology
from dispersion import *
import matplotlib.animation as animation
import registration
import SimpleITK as sitk

from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread)
import pynvml,cv2
import time,os
import sys
import numpy as np
import scipy.io
import cupyx.scipy.signal

from PIL import Image,ImageEnhance,ImageOps,ImageQt
from octFuncs import *
from random import randint
from sidebar_ui import Ui_MainWindow
#%load_ext memory_profiler

class TSAThread(QThread):
    volume_ready = Signal(np.ndarray)
    def __init__(self, processParameters, frame_3D, frame_OCTA):
        self.processParameters = processParameters
        self.frame_3D = frame_3D
        self.frame_OCTA = frame_OCTA
        super().__init__()

    def demons_registration(self,fixed_image, moving_image):
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(2)
        displacementField = demons.Execute(fixed_image, moving_image)
        outTx = sitk.DisplacementFieldTransform(displacementField)
        return outTx


    def demons_transformation(self,fixed_image, moving_image, displacement):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(displacement)
        output = resampler.Execute(moving_image)
        #output_img = sitk.Cast(sitk.RescaleIntensity(output), sitk.sitkUInt8)
        output_array = sitk.GetArrayFromImage(output)
        return output_array


    def smooth(self,a,WSZ=5):
        # a: NumPy 1-D array containing the data to be smoothed
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = cp.convolve(a,cp.ones(WSZ,dtype=int),'valid')/WSZ    
        r = cp.arange(1,WSZ-1,2)
        start = cp.cumsum(a[:WSZ-1])[::2]/r
        stop = (cp.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        return cp.concatenate((  start , out0, stop  ))


    def aline_cross_correlation(self, tempSpectraBscan, aline):
        aline1 = cp.fft.fft(aline)
        shiftInf = cp.zeros((cp.shape(tempSpectraBscan)[1],1))
        for i in range(cp.shape(tempSpectraBscan)[1]):
            aline2 = cp.fft.fft(cp.squeeze(tempSpectraBscan[:,i]))
            crs_cor = cp.abs(cp.fft.ifft(aline1*cp.conj(aline2)))
            shift = cp.argmax(crs_cor)
            tempSpectraBscan[:,i] = cp.roll(tempSpectraBscan[:,i],shift)
            shiftInf[i] = shift
        return shiftInf

    def run(self):
        frame_3D_out = self.frame_3D
        frame_OCTA = self.frame_OCTA
        tsa_frame_3D = [frame_3D_out,frame_3D_out,frame_3D_out,frame_3D_out,frame_3D_out]
        tsa_frame_OCTA = [frame_OCTA,frame_OCTA,frame_OCTA,frame_OCTA,frame_OCTA]
        processParameters = self.processParameters

        wavePath = 'Wavelength Files/wavelength-OH.01.2021.0005-created-28-Jan-2021'
        with open(wavePath, 'rb') as f:
            b = f.read()

        wavelength = np.frombuffer(b)
        wavelength = wavelength*1e-9

        linear_wavenumber_matrix = linear_k()
        linear_wavenumber_matrix.calculate_linear_interpolation_matrix(wavelength,processParameters)

        class stft:
            pass
        stft.winNum = 24
        stft.resol = 11e-6
        stft.firstWave = 522e-9
        stft.lastWave = 592e-9
        stft = calculateWindows(stft,wavelength,processParameters,processParameters.pixelMap)
        stft.stft_w_g = cp.asarray(stft.stftWin)

        linear_wavenumber_matrix.To_GPU()
        del frame_3D_out
        del frame_OCTA
        del self.frame_3D
        del self.frame_OCTA
        for i in range(4):
            raw_fringes = fringes(processParameters.fname,processParameters,processParameters.pixelMap,i+1)
            raw = raw_fringes.get_balance_Fringes()
            tsa_frame_3D[i+1],tsa_frame_OCTA[i+1] = octangio_Recon_Bal_GPU(raw,cp.array(processParameters.dispersion),stft.stft_w_g,processParameters,linear_wavenumber_matrix.kmap)
            
        morph_ellipse = morphology.disk(15)
        morph_ellipse = morph_ellipse[1:-1,1:-1]

        octa_enfaces = cp.zeros((512,512,5))
        enfaces = np.zeros((512,512,5))
        for i in range(5):
            enface = np.squeeze(np.mean(tsa_frame_3D[i],0))
            enface = enface/np.mean(enface)
            enface = enface/np.max(enface)
            enfaces[:,:,i] = enface
            
            enface0 = np.squeeze(np.mean(tsa_frame_OCTA[i],0))
            enface0 = enface0/np.mean(enface0)
            enface0 = enface0**3
            enface0 = enface0 - cv2.morphologyEx(enface0, cv2.MORPH_OPEN, morph_ellipse)
            enface0 += abs(np.min(enface0))
            enface0 = enface0/np.max(enface0)
            octa_enfaces[:,:,i] = cp.asarray(enface0)

        dispMatrix = []
        v_shift = []
        h_shift = []
        octa_ref_enface = octa_enfaces[:,:,1]
        octa_ref_sitk = sitk.GetImageFromArray(octa_ref_enface.get())
        for i in range(5):
            #rigid transformation
            out = register_images(octa_ref_enface, octa_enfaces[:,:,i],usfac = 1)
            vershift = int(out[1])
            horshift = int(out[0])
            v_shift.append(vershift)
            h_shift.append(horshift)
            enfaces[:,:,i] = applyBulkAlignment_np(enfaces[:,:,i],vershift,horshift)
            octa_enfaces[:,:,i] = applyBulkAlignment(octa_enfaces[:,:,i],vershift,horshift)
            plt.imshow(abs(octa_enfaces[i]).get(),cmap='gray')
            
            #non-rigid demon registration
            octa_cur_sitk = sitk.GetImageFromArray(octa_enfaces[:,:,i].get())
            out = self.demons_registration(octa_ref_sitk,octa_cur_sitk)
            dispMatrix.append(out)
            
            enface_sitk = sitk.GetImageFromArray(enfaces[:,:,i])
            enface_transformed = self.demons_transformation(enface_sitk,enface_sitk,out)
            enfaces[:,:,i] = enface_transformed

            octa_enface_transformed = self.demons_transformation(octa_ref_sitk,octa_cur_sitk,out)
            octa_enfaces[:,:,i] = cp.asarray(octa_enface_transformed)
            

        out_3D = tsa_frame_3D.copy()
        out_OCTA = tsa_frame_OCTA.copy()
        for i in range(5):
            for depth in range(np.shape(tsa_frame_3D[i])[0]):
                tsa_frame_3D[i][depth,:,:] = applyBulkAlignment_np(tsa_frame_3D[i][depth,:,:],v_shift[i],h_shift[i])
                tsa_frame_OCTA[i][depth,:,:] = applyBulkAlignment_np(tsa_frame_OCTA[i][depth,:,:],v_shift[i],h_shift[i])
                
                frame = np.flipud(cp.asnumpy(out_3D[0][depth,:,:]))

                frame_sitk = sitk.GetImageFromArray(tsa_frame_3D[i][depth,:,:])
                ref_frame_sitk = sitk.GetImageFromArray(tsa_frame_3D[1][depth,:,:])
                out_3D[i][depth,:,:] = self.demons_transformation(ref_frame_sitk,frame_sitk,dispMatrix[i])
                
                frame = np.flipud(cp.asnumpy(out_3D[0][depth,:,:]))
                
                OCTA_frame_sitk = sitk.GetImageFromArray(tsa_frame_OCTA[i][depth,:,:])
                ref_OCTA_frame_sitk = sitk.GetImageFromArray(tsa_frame_OCTA[1][depth,:,:])
                out_OCTA[i][depth,:,:] = self.demons_transformation(ref_OCTA_frame_sitk,OCTA_frame_sitk,dispMatrix[i])

        frame_3D_avg = np.zeros((1024,512,512))
        frame_OCTA_avg = np.zeros((1024,512,512))
        upSampleFactor = 6
            
        startTime = time.time()

        for i in range(512):

            bscan_3D = cp.zeros((1024,512,5))
            bscan_OCTA = cp.zeros((1024,512,5))
            
            print("bscan number:",i)

            for vol in range(5):
                bscan_3D[:,:,vol] = cp.asarray(out_3D[vol][:,:,i])
                bscan_OCTA[:,:,vol] = cp.asarray(out_OCTA[vol][:,:,i])

            refNum = 2
            refBscan = bscan_3D[:,:,refNum]
            
            for vol in range(5):
                curBscan = bscan_3D[:,:,vol]
                shift = register_images(cp.asarray(refBscan), cp.asarray(curBscan),usfac = 1)

                bscan_3D[:,:vol] = applyBulkAlignment(bscan_3D[:,:vol],int(shift[1]),int(shift[0]))
                bscan_OCTA[:,:,vol] = applyBulkAlignment(bscan_OCTA[:,:,vol],int(shift[1]),int(shift[0]))
            

            bscan_3D_upsample = cp.zeros((cp.shape(bscan_3D)[0]*upSampleFactor,cp.shape(bscan_3D)[1],cp.shape(bscan_3D)[2]))
            bscan_OCTA_upsample = cp.zeros((cp.shape(bscan_OCTA)[0]*upSampleFactor,cp.shape(bscan_OCTA)[1],cp.shape(bscan_OCTA)[2]))
            for vol in range(cp.shape(bscan_3D)[2]):
                for aline_number in range(cp.shape(bscan_3D)[1]):
                    bscan_3D_upsample[:,aline_number,vol] = cp.interp(cp.linspace(0,cp.shape(bscan_3D)[0]-1,num=cp.shape(bscan_3D)[0]*upSampleFactor),cp.linspace(0,cp.shape(bscan_3D)[0]-1,num=cp.shape(bscan_3D)[0]),bscan_3D[:,aline_number,vol])
                    bscan_OCTA_upsample[:,aline_number,vol] = cp.interp(cp.linspace(0,cp.shape(bscan_OCTA)[0]-1,num=cp.shape(bscan_OCTA)[0]*upSampleFactor),cp.linspace(0,cp.shape(bscan_OCTA)[0]-1,num=cp.shape(bscan_OCTA)[0]),bscan_OCTA[:,aline_number,vol])

            yc = []
            shiftMats = cp.zeros((processParameters.res_fast,5))
            
            top = cp.mean(bscan_3D_upsample[0:4000,int(processParameters.res_fast/2),refNum])
            bot = cp.mean(bscan_3D_upsample[-4000:,int(processParameters.res_fast/2),refNum])
            if top > bot:
                cropTop = 0
                cropBot = 4000
            else:
                cropTop = 2144
                cropBot = 6144
            for aline_number in range(processParameters.res_fast):

                refLine = bscan_3D_upsample[cropTop:cropBot,aline_number,refNum]

                    #print(cp.shape(curFrame_stack_upsample[cropTop:cropBot,aline_number,:]))
                compLines = cp.zeros(cp.shape(bscan_3D_upsample[cropTop:cropBot,aline_number,:]))
                for vol in range(5):
                    compLines[:,vol] = self.smooth(bscan_3D_upsample[cropTop:cropBot,aline_number,vol])


                aligned_alines = cp.squeeze(compLines)
                shiftAmt = self.aline_cross_correlation(aligned_alines,refLine)
                
                #print(shiftAmt[:,0])
                
                #break
                for vol in range(5):
                    cur_shift = shiftAmt[vol]
                    if abs(cur_shift) > 100*processParams.upSampleFactor:
                        cur_shift = cur_shift - cp.shape(aligned_alines)[0] if (cur_shift > 0) else cur_shift + cp.shape(aligned_alines)[0]
                    shiftMats[aline_number,vol] = cur_shift

                    
                    
            for vol in range(5):
                shiftMats[:,vol] = cp.round(cupyx.scipy.signal.medfilt(shiftMats[:,vol],15))
            #print(shiftMats[:,0])

            
            for vol in range(5):
                for aline_number in range(cp.shape(bscan_3D)[1]):
                        # check if cp.roll should be negative
                    bscan_3D[:,aline_number,vol] = cp.interp(cp.linspace(0,cp.shape(bscan_3D_upsample)[0]-1,num=cp.shape(bscan_3D)[0]),cp.linspace(0,cp.shape(bscan_3D_upsample)[0]-1,num=cp.shape(bscan_3D_upsample)[0]),cp.roll(bscan_3D_upsample[:,aline_number,vol],int(shiftMats[aline_number,vol])))
                    bscan_OCTA[:,aline_number,vol] = cp.interp(cp.linspace(0,cp.shape(bscan_OCTA_upsample)[0]-1,num=cp.shape(bscan_OCTA)[0]),cp.linspace(0,cp.shape(bscan_OCTA_upsample)[0]-1,num=cp.shape(bscan_OCTA_upsample)[0]),cp.roll(bscan_OCTA_upsample[:,aline_number,vol],int(shiftMats[aline_number,vol])))
            
            
            print("after aline registration")
            frame_3D_avg[:,:,i] = cp.squeeze(cp.mean(bscan_3D,axis=2)).get()
            frame_OCTA_avg[:,:,i] = cp.squeeze(cp.mean(bscan_OCTA,axis=2)).get()
            

        endTime = time.time()
        print("Time used:",(endTime-startTime))
        self.volume_ready.emit(frame_3D_avg,frame_OCTA_avg)