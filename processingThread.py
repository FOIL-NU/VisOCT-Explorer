from PySide6.QtCore import (Qt,QFile, QTextStream,QObject, Signal, QDir,QThread,QTimer,QEvent,QPoint,QRectF)

from balancefringe import fringes
from processParams import processParams
from resample import linear_k
from skimage import exposure,morphology
from skimage.transform import resize
from dispersion import *
import matplotlib.animation as animation
import registration
import time,os,gc
import sys,cv2
import numpy as np

from octFuncs import *
from random import randint
from sidebar_ui import Ui_MainWindow
from sklearn.mixture import GaussianMixture

from scipy.signal import hilbert


class ProcessingThread(QThread):
    finished = Signal()
    progress = Signal(int,float)
    image_ready = Signal(np.ndarray,str,float,float,np.ndarray,np.ndarray,float,float)
    updateProgressBar = Signal(int)

    def __init__(self, batch=False,match_path=None):
        self.batch = batch
        self.match_path = match_path
        super().__init__()

    def set_attrib(self,processParameters,preview_mode=False,preset_dispersion=None,preview_bscans=32):
        self.processParameters = processParameters
        self.preview_mode = preview_mode
        self.preset_dispersion = preset_dispersion
        self.preview_bscans = preview_bscans

    def imageQI_3D(self,frame_3D_QI, val_Low, val_Noise, val_Saturation):
        frame_3D_QI = 20*np.log10(frame_3D_QI)
        CC = len(frame_3D_QI[0, 0, :])
        val_Middle = (val_Noise + val_Saturation) / 2
        img3D1 = np.multiply(frame_3D_QI >= val_Middle, frame_3D_QI <= val_Saturation)
        img3D2 = np.multiply(frame_3D_QI >= val_Noise, frame_3D_QI <= val_Middle)
        IR = (val_Saturation - val_Low) / np.abs(val_Low) * 100
        print("IR:",IR)
        TSR = np.zeros((CC, 1))
        for ccc in range(0, CC):
            imgTemp1 = img3D1[:, :, ccc]
            imgTemp2 = img3D2[:, :, ccc]
            imgTemp1 = np.where(imgTemp1 == 1)
            imgTemp2 = np.where(imgTemp2 == 1)
            if np.size(imgTemp2) == 0:
                TSR[ccc] = 0
            else:
                TSR[ccc] = np.size(imgTemp1) / np.size(imgTemp2)
        QI = IR * TSR
        return QI

    def imgQI(self,imgB, defLow, defNoise, defSaturation):
        imgB = 20*np.log10(imgB)
        if defLow > 0:   # self-defined thresholds
            val_Low = defLow
            val_Noise = defNoise
            val_Saturation = defSaturation
        else:   # paper threshold
            val_Low = np.percentile(imgB, 1)  # 1st percentile
            val_Noise = np.percentile(imgB, 70)  # 75th percentile
            val_Saturation = np.percentile(imgB, 99)  # 99th percentile
        val_Middle = (val_Noise + val_Saturation) / 2
        print("middle:",val_Middle)
        print("sat:",val_Saturation)
        imgTemp1 = np.multiply(imgB >= val_Middle, imgB <= val_Saturation)
        imgTemp2 = np.multiply(imgB >= val_Noise, imgB <= val_Middle)
        imgTemp1 = np.where(imgTemp1 == 1)
        imgTemp2 = np.where(imgTemp2 == 1)
        IR = (val_Saturation - val_Low) / val_Low * 100
        if np.size(imgTemp2) == 0:
            TSR = 0
        else:
            TSR = np.size(imgTemp1) / np.size(imgTemp2)
        QI = IR * TSR
        return QI

    def generate_enface(self,frame_3D_resh,octa,directory,aspect,frame_OCTA=None):
        if GPU_available:
            enface = cp.squeeze(cp.mean(frame_3D_resh,axis=0))
            print("enface shape",cp.shape(enface))
            #enface = cp.asnumpy(enface)
            #print("max:",np.max(enface))
            #enface = ((np.clip((enface - 5) / (np.max(enface)-225000), 0, 1)) * 255).astype(np.uint8)
            enface = enface/cp.mean(enface,axis=0)
            enface = cp.asnumpy(enface/cp.max(enface))
            #enface = exposure.equalize_adapthist(enface,clip_limit=0.01)
        else:
            enface = np.squeeze(np.mean(frame_3D_resh,axis=0))
            print("enface shape",np.shape(enface))
            enface = enface/np.mean(enface,axis=0)
            enface = enface/np.max(enface)
        if octa:
            #frame_OCTA = resize(frame_OCTA,(1024,1024,np.shape(frame_OCTA)[2]))
            enface = enface/np.max(enface)
            octa_enface = np.squeeze(np.mean(frame_OCTA,axis=0))
            octa_enface = octa_enface/np.mean(octa_enface)
            octa_enface = octa_enface**3
            morph_ellipse = morphology.disk(15)
            morph_ellipse = morph_ellipse[1:-1,1:-1]
            octa_enface = octa_enface - cv2.morphologyEx(octa_enface, cv2.MORPH_OPEN, morph_ellipse)
            octa_enface += abs(np.min(octa_enface))
            octa_enface = octa_enface/np.max(octa_enface)

            enface = exposure.equalize_adapthist(enface,clip_limit=0.008)
            octa_enface = exposure.equalize_adapthist(octa_enface,clip_limit=0.008)
            #enface = ((np.clip((enface - 0.02) / (0.4 - 0.02), 0, 1)) * 255).astype(np.uint8)
            plt.imsave(directory+'\octa_enface.tiff', octa_enface, cmap='gray') 
            #enface = enface/np.mean(enface,0)
        else:

            enface = exposure.equalize_adapthist(enface,clip_limit=0.003)
        enface_size = min(enface.shape[0],enface.shape[1])
        enface = cv2.resize(enface,(enface_size,enface_size),interpolation=cv2.INTER_LINEAR)
        plt.imsave(directory+'\enface.tiff', enface, cmap='gray')
        return enface
    
    def find_low_high(self,directory,frame,is_octa=False):
        data = []
        hist, bins = cp.histogram(frame,bins=200)
        hist = hist.get()
        bins = bins.get()
        for count, bin_start, bin_end in zip(hist, bins[:-1], bins[1:]):
            data.extend(np.random.uniform(bin_start, bin_end, count))
        data = np.array(data).reshape(-1, 1)

        lowVal = cp.mean(frame)+1
        highVal = lowVal+2.25*cp.std(frame)+1

        hist = hist/(np.max(hist)*0.05)
        plt.clf()
        plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
        #print(1)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlim(0, 100)
        plt.gca().set_aspect(1)
        plt.grid(True)
        plt.savefig(directory+"\log_scale_contrast.tiff", dpi=300, bbox_inches='tight', pad_inches=0)
                #print(2)
        
        return [lowVal,highVal]

    def find_low_high_octa(self,directory,frame,is_octa=True):
        flattened_img = frame.flatten()

        # Plot the histogram
        value_range = (0, 4096)

        # Plot the histogram with the specified range

        plt.clf()
        plt.figure(figsize=(8, 2))
        plt.hist(flattened_img, bins=100, range=value_range, color='blue', alpha=0.7)


        #plt.clf()
        #plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlim(0, 4096)
        #plt.gca().set_aspect(1)
        plt.grid(True)

        lowVal = np.mean(flattened_img)-0.5*np.std(flattened_img)
        highVal = np.mean(flattened_img)+0.4*np.std(flattened_img)

        plt.savefig(directory+"\log_scale_octa_contrast.tiff", dpi=300, bbox_inches='tight', pad_inches=0)
        return [lowVal,highVal]



    def run(self):
        print("at start processing")
        if self.batch:
            num_dataset = len(self.processParameters)
        else:
            num_dataset = 1
        for i in range(num_dataset):
            if self.batch:
                processParameters = processParams(32,self.processParameters[i],self.match_path)
            else:
                processParameters = self.processParameters
        
            raw_fringes = fringes(processParameters.fname,processParameters,processParameters.pixelMap)
            
            class data:
                pass
            wavePath = processParameters.wavelength_file
            with open(wavePath, 'rb') as f:
                b = f.read()

            data.wavelength = np.frombuffer(b)
            data.wavelength = data.wavelength*1e-9
            self.progress.emit(0,0)
            startTime = time.time()
            if processParameters.balFlag:
                print("at start balancing")
                data.raw = raw_fringes.get_balance_Fringes(progress=self.updateProgressBar)
                
            else:
                data.raw = raw_fringes.get_unbalance_Fringes(progress=self.updateProgressBar)
            endTime = time.time()
            del raw_fringes

            if self.processParameters.envelop:

                envelop = np.zeros_like(data.raw)
                for i in range(np.shape(data.raw)[0]):
                    aline = data.raw[i,:]
                    envelop[i,:] = np.abs(hilbert(aline))
                envelop_avg = np.mean(envelop,axis=0)
                data.raw = data.raw/envelop_avg
                del envelop

            gc.collect()

            self.progress.emit(1,round(endTime-startTime,2))

            linear_wavenumber_matrix = linear_k()

            linear_wavenumber_matrix.calculate_linear_interpolation_matrix(data.wavelength,processParameters)
            self.updateProgressBar.emit(2)
            self.progress.emit(2,0)

            class stft:
                pass

            stft.winNum = 24
            stft.resol = 11e-6
            stft.firstWave = 522e-9
            stft.lastWave = 592e-9

            stft = calculateWindows(stft,data.wavelength,processParameters,processParameters.pixelMap)

            #plt.plot(stft.stftWin)
            
            # %% Dispersion Compensation
            self.progress.emit(3,0)
            class dispersion:
                pass
            startTime = time.time()
            if GPU_available:
                stft.stft_w_g = cp.asarray(stft.stftWin)
                linear_wavenumber_matrix.To_GPU()
            #linear_wavenumber_matrix.To_Torch()
            if not self.preview_mode:
                dispersion.c2a, dispersion.c3a = dispersionOptimization_balance(data.raw,data.wavelength,processParameters,processParameters.pixelMap,linear_wavenumber_matrix.kmap,0,self.updateProgressBar)
            else:
                dispersion.c2a = self.preset_dispersion
                dispersion.c3a = 0
            #dispersion.c2a = 5.68
            #dispersion.c3a  = -0.0408

            dispersion.dispersion = calcDispersionPhase(linear_wavenumber_matrix.kmap.freq_lin,dispersion.c2a,dispersion.c3a)
            processParameters.dispersion = dispersion.dispersion
            endTime = time.time()
            self.progress.emit(4,round(endTime-startTime,2))
            # %% Basic OCT reconstruction
            
            self.progress.emit(5,0)
            startTime = time.time()
            frame_OCTA = None
            octa_lowVal = None
            octa_highVal = None
            
            if GPU_available and (cp.cuda.runtime.getDeviceProperties(cp.cuda.Device())["totalGlobalMem"] >= 2147483648):
                #Total memory > 6GB
                if processParameters.octaFlag:
                    frame_3D_out,frame_OCTA = octangio_Recon_Bal_GPU(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g,processParameters,linear_wavenumber_matrix.kmap)
                else:
                    #frame_3D_out = octRecon_Bal_GPU(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g[:,0],processParameters,512*2,linear_wavenumber_matrix.kmap)
                    frame_3D_out = octRecon_Bal_GPU(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g[:,0],processParameters,512*2,linear_wavenumber_matrix.kmap)
                cp.get_default_memory_pool().free_all_blocks()
            else:
                frame_3D_out = octRecon_Bal(data.raw,dispersion.dispersion,stft.stftWin[:,0],processParameters,512*4,linear_wavenumber_matrix.kmap)
            endTime = time.time()

            del data.raw
            self.updateProgressBar.emit(4)
            self.progress.emit(6,round(endTime-startTime,2))
            if self.preview_mode:
                frame_3D_resh = np.reshape(frame_3D_out,(int(processParameters.res_axis/2),int(processParameters.res_fast),self.preview_bscans),'A')
            else:
                frame_3D_resh = np.reshape(frame_3D_out,(int(processParameters.res_axis/2),int(processParameters.res_fast),int(processParameters.res_slow)),'A')
            # Flip every other frame
            self.progress.emit(7,0)
            backLash = 0
            directory = processParameters.fname.split('.RAW')[0]
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            xpw = processParameters.xrng/processParameters.res_fast        
            ypw = processParameters.yrng/(processParameters.res_slow)
            zpw = processParameters.zrng/(processParameters.res_axis/2)
            if not processParameters.SRFlag:
                if not processParameters.octaFlag:
                    
                    if GPU_available:
                        if processParameters.res_fast!=8192:
                            print("process parameter res fast != 8192")
                            registration.horizontal_flip(frame_3D_resh)
                            startTime = time.time()
                            
                            #frame_3D_resh_GPU= cp.asarray(frame_3D_resh[:,:,:int(processParameters.res_slow/2)])
                            frame_3D_resh_GPU = cp.asarray(frame_3D_resh)
                            
                            vert_shift = [0]
                            if not self.batch:
                                backLash = registration.get_backlash_pattern(frame_3D_resh_GPU,int(processParameters.res_slow),vert_shift,self.updateProgressBar)
                            else:
                                backLash = registration.get_backlash_pattern(frame_3D_resh_GPU,int(processParameters.res_slow),vert_shift)
                            del frame_3D_resh_GPU
                            registration.horizontal_correction(frame_3D_resh,processParameters.res_slow,backLash)               
                            print("backlash is:",backLash)
                            frame_3D_resh = frame_3D_resh[:,:-np.abs(backLash),:]
                            if self.preview_mode != True:
                                enface = self.generate_enface(frame_3D_resh,False,directory,xpw/ypw)
                            chunk_num = 2
                            per_chunk = int(processParameters.res_slow/chunk_num)                   
                            temp = cp.zeros((cp.shape(frame_3D_resh)[0],cp.shape(frame_3D_resh)[1],int(cp.shape(frame_3D_resh)[2]/4)))
                            cur_shift_count = 0

                            for cumulative_vertical_shift in vert_shift:
                                frame_3D_resh[:,:,cur_shift_count] = np.roll(frame_3D_resh[:,:,cur_shift_count],-int(cumulative_vertical_shift),0)
                                cur_shift_count += 1 

                            for i in range(chunk_num):
                                to_GPU = registration.shift_correction_bulk(frame_3D_resh[:,:,i*per_chunk:(i+1)*per_chunk],4)
                                for frame_number in range(int(cp.shape(to_GPU)[2]/4)):
                                    temp[:,:,i*(per_chunk/4)+frame_number] = cp.squeeze(cp.mean(to_GPU[:,:,(frame_number*4):(frame_number*4+4)],axis=2))
                                del to_GPU
                                cp.get_default_memory_pool().free_all_blocks()
                        
                        else:
                            print("process parameter res fast == 8192")
                            registration.horizontal_flip(frame_3D_resh)

                            frame_3D_resh_GPU = cp.asarray(frame_3D_resh)
                            if self.preview_mode != True:
                                enface = self.generate_enface(frame_3D_resh,False,directory,xpw/ypw)
                            temp = cp.zeros((1024,1024,16))
                            for i in range(0,1024):
                                temp[:,i,:] = cp.squeeze(cp.mean(frame_3D_resh_GPU[:,i*8:i*8+8,:],axis=1))
                            #temp = cp.asarray(frame_3D_resh)
                        endTime = time.time()
                        self.progress.emit(8,round(endTime-startTime,2))
                        
                        #frame = 20*cp.log10(temp[:,:,np.shape(temp)[2]/2])
                        contrast_volume = 20*cp.log10(temp[:,:,int(np.shape(temp)[2]/2)-8:int(np.shape(temp)[2]/2)+8])
                        #contrast_volume = 20*np.log10(frame_3D_resh[:,:,99])
                        self.progress.emit(17,round(0))
                        lowVal,highVal = self.find_low_high(directory,contrast_volume)
                        self.progress.emit(18,round(0))
                    else:
                        registration.horizontal_flip(frame_3D_resh)
                        startTime = time.time()
                        
                        backLash = registration.get_backlash_pattern(frame_3D_resh,int(processParameters.res_slow/2),self.updateProgressBar)
                        registration.horizontal_correction(frame_3D_resh,processParameters.res_slow,backLash)
                        
                        print("backlash is:",backLash)

                        frame_3D_resh = frame_3D_resh[:,:-np.abs(backLash),:]
                        if self.preview_mode != True:
                            enface = self.generate_enface(frame_3D_resh,False,directory,xpw/ypw)
                        temp = np.zeros((np.shape(frame_3D_resh)[0],np.shape(frame_3D_resh)[1],int(np.shape(frame_3D_resh)[2]/4)))
                        registration.vertical_correction(frame_3D_resh,int(processParameters.res_slow))
                        corrected = registration.shift_correction_bulk(frame_3D_resh,4)
                            
                        for frame_number in range(int(np.shape(corrected)[2]/4)):
                            temp[:,:,i*(per_chunk/4)+frame_number] = np.squeeze(np.mean(corrected[:,:,(frame_number*4):(frame_number*4+4)],axis=2))

                        endTime = time.time()
                        self.progress.emit(8,round(endTime-startTime,2))

                        frame = 20*cp.log10(temp[:,:,np.shape(temp)[2]/2])
                        contrast_volume = 20*cp.log10(temp[:,:,np.shape(temp)[2]/2-16:np.shape(temp)[2]/2+16])
                        self.progress.emit(17,round(0))
                        lowVal,highVal = self.find_low_high(directory,contrast_volume)
                        self.progress.emit(18,round(0))

                else:
                    contrast_volume = (frame_OCTA[:,:,int(np.shape(frame_OCTA)[2]/2)-16:int(np.shape(frame_OCTA)[2]/2)+16])
                    octa_lowVal,octa_highVal = self.find_low_high_octa(directory,contrast_volume,True)
                    octa_lowVal = 0
                    octa_highVal = 10*np.mean(frame_OCTA)
                    print("octa lowVal is:", octa_lowVal)
                    print("octa highVal is:", octa_highVal)
                    if self.preview_mode != True:
                        enface = self.generate_enface(frame_3D_resh,True,directory,xpw/ypw,frame_OCTA)
                    
                    contrast_volume = 20*cp.log10(cp.asarray(frame_3D_resh[:,:,int(np.shape(frame_3D_resh)[2]/2)-4:int(np.shape(frame_3D_resh)[2]/2)+4]))
                    print(4)
                    self.progress.emit(17,round(0))
                    lowVal,highVal = self.find_low_high(directory,contrast_volume)
                    self.progress.emit(18,round(0))
                cache = cp.fft.config.get_plan_cache()
                cache.clear()

                print("lowVal is:", lowVal)
                print("highVal is:", highVal)
                print("processParameters.xrng is:",processParameters.xrng)

                
            else:
                if processParameters.newSRFlag:
                    human_SR = False
                    frame_3D_resh_resize = cp.zeros((int(processParameters.res_axis/2),int(processParameters.res_fast),int(processParameters.totalBnum)))
                    
                    for i in range(processParameters.totalBnum):

                        curFrame_stack = frame_3D_resh[:,:,i*processParameters.numAvgs:(i+1)*processParameters.numAvgs]
                        print(processParameters.totalBnum)
                        for frame_number in range(processParameters.numAvgs):
                            print(processParameters.numAvgs)
                            if frame_number%2 == 1:
                                curFrame_stack[:,:,frame_number] = cp.fliplr(curFrame_stack[:,:,frame_number])
                        if i == 0:
                            backLash = registration.get_backlash_pattern(curFrame_stack,processParameters.numAvgs)
                            print("backLash", backLash)
                        registration.horizontal_correction(curFrame_stack,processParameters.numAvgs,backLash)

                        if human_SR:
                            curFrame_stack_upsample = cp.zeros((cp.shape(curFrame_stack)[0]*upSampleFactor,cp.shape(curFrame_stack)[1],cp.shape(curFrame_stack)[2]))
                            for frame_number in range(cp.shape(curFrame_stack)[2]):
                                for aline_number in range(cp.shape(curFrame_stack)[1]):
                                    curFrame_stack_upsample[:,aline_number,frame_number] = cp.interp(cp.linspace(0,cp.shape(curFrame_stack)[0]-1,num=cp.shape(curFrame_stack)[0]*upSampleFactor),cp.linspace(0,cp.shape(curFrame_stack)[0]-1,num=cp.shape(curFrame_stack)[0]),curFrame_stack[:,aline_number,frame_number])

                        else:
                            curFrame_stack_upsample = curFrame_stack
                        

                        #Where is generate enface?

                        refNum = int(processParameters.numAvgs/2)
                        refBscan = curFrame_stack_upsample[:,:,refNum]
                        
                        for frame_number in range(processParameters.numAvgs):
                            curBscan = curFrame_stack_upsample[:,:,frame_number]
                            shift = register_images(refBscan, curBscan,usfac = 1)
                        #    #print(cp.asnumpy(shift[0]))
                            curFrame_stack_upsample[:,:,frame_number] = applyBulkAlignment(curBscan,int(shift[1]),int(shift[0]))
                        #temp1 = cp.mean(curFrame_stack[:,:,0:32],axis=2)
                        frame_3D_resh_resize[:,:,i] = cp.squeeze(cp.mean(curFrame_stack_upsample,axis=2))

                    frame_3D_resh = frame_3D_resh_resize
                    lowVal = 0
                    #lowVal = float(cp.mean(20*cp.log10(frame_3D_resh))+1)
                    highVal = float(lowVal+1.75*cp.std(20*cp.log10(frame_3D_resh)))
                    xpw = processParameters.xrng/processParameters.res_fast
                    ypw = processParameters.yrng/processParameters.totalBnum
                    zpw = processParameters.zrng/(processParameters.res_axis/2)
                else:
                    if processParameters.res_fast == 8192:
                        frame_3D_resh_resize = np.zeros((frame_3D_resh.shape[0],int(processParameters.res_fast/8),int(processParameters.res_slow)))

                        for i in range(0,processParameters.res_fast,8):
                            temp = frame_3D_resh[:,i:i+8,:]
                            #print(temp.shape)
                            frame_3D_resh_resize[:,int(i/8),:] = np.mean(temp,axis=1)
                        frame_3D_resh = frame_3D_resh_resize
                    
                    for i in range(0,frame_3D_resh.shape[2],2):
                        frame_3D_resh[:,:,i] = np.fliplr(frame_3D_resh[:,:,i])
                    frame_3D_resh_GPU = cp.asarray(frame_3D_resh)
                    vert_shift = [0]
                    backLash = registration.get_backlash_pattern(frame_3D_resh_GPU,int(processParameters.res_slow),vert_shift,self.updateProgressBar)
                    frame_3D_resh_GPU
                    registration.horizontal_correction(frame_3D_resh,processParameters.res_slow,backLash)               
                    print("backlash is:",backLash)
                    frame_3D_resh = frame_3D_resh[:,:-np.abs(backLash),:]
                    xpw = processParameters.xrng/1024
                    ypw = processParameters.yrng/processParameters.res_slow
                    zpw = processParameters.zrng/(processParameters.res_axis/2)
                    contrast_volume = 20*cp.log10(frame_3D_resh_GPU[:,:,int(np.shape(frame_3D_resh)[2]/2)-4:int(np.shape(frame_3D_resh)[2]/2)+4])
                    del frame_3D_resh_GPU
                    self.progress.emit(17,round(0))
                    lowVal,highVal = self.find_low_high(directory,contrast_volume)
                    self.progress.emit(18,round(0))

                if self.preview_mode != True:
                    enface = self.generate_enface(frame_3D_resh,False,directory,xpw/ypw)
                


            if processParameters.SRFlag and not processParameters.newSRFlag:
                frame_3D_resh = 20*np.log10(cp.asnumpy(frame_3D_resh))
            

            #plt.imshow(20*np.log10(enface),cmap='gray',vmin = 9, vmax = 23, aspect = ear)
            temp_QI = 0
            def_low = 22
            frame_3D_resh_log = 20*np.log10(frame_3D_resh)
            def_low = np.mean(frame_3D_resh_log) - 2.3*np.std(frame_3D_resh_log)
            def_noise = np.mean(frame_3D_resh_log) + 1.8*np.std(frame_3D_resh_log)
            def_Saturation = np.mean(frame_3D_resh_log) + 2.3*np.std(frame_3D_resh_log)
            
            temp_QI = self.imageQI_3D(frame_3D_resh,def_low,def_noise,def_Saturation)
            
            if self.batch:
                bar_f = xpw/zpw
                average_number = 16
                save_mat = cp.asarray(frame_3D_resh)
                start_frame_num = processParameters.res_slow/(32*2)-1
                increment = processParameters.res_slow/32
                registration.shift_correction_bulk(save_mat,average_number)
                print("bar_f:",bar_f)
                
                if not os.path.exists(directory+"/"+str(average_number)+"_averaged"):
                    os.makedirs(directory+"/"+str(average_number)+"_averaged")
                for i in range(32):
                    averaged_img = cp.squeeze(cp.mean(save_mat[:,:,start_frame_num+i*increment-average_number/2:min(start_frame_num+i*increment+average_number/2,processParameters.res_slow-1)],axis=2))
                    #plt.imshow(20*np.log10(averaged_img.get()), cmap = 'gray', origin = "lower", aspect = bar_f, vmin = self.lowVal, vmax = self.highVal)
                    #plt.savefig(self.directory+'/OCT_Reconstruct_'+str(average_number)+'_averaged_'+str(i)+'.tiff')
                    image = cp.asnumpy((cp.flipud((cp.clip((averaged_img - lowVal) / (highVal - lowVal), 0, 1)) * 255)).astype(cp.uint8))
                    cv2_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    res = cv2.resize(cv2_image, dsize=(4*int(np.shape(image)[0]*bar_f), 4*np.shape(image)[0]))
                    cv2.imwrite(directory+"/"+str(average_number)+'_averaged/OCT_Reconstruct_'+str(average_number)+'_averaged_'+str(i)+'.tiff', res)
            else:
                if not processParameters.SRFlag and processParameters.balFlag and not processParameters.octaFlag:
                    #normal oct image
                    temp = np.float32(temp.get())
                    cp.get_default_memory_pool().free_all_blocks()
                    if self.preview_mode != True:
                        np.save(directory+'/frame_3D.npy',frame_3D_resh)
                    self.image_ready.emit(temp,directory,lowVal,highVal,temp_QI,frame_OCTA,octa_lowVal,octa_highVal)
                
                    del temp
                elif processParameters.octaFlag:
                    #frame_3D_resh_20log = 20*np.log10(frame_3D_resh)
                    #frame_3D_resh = np.float32(frame_3D_resh)
                    if self.preview_mode != True:
                        np.save(directory+'/frame_3D.npy',frame_3D_resh)
                    self.image_ready.emit(frame_3D_resh,directory,lowVal,highVal,temp_QI,frame_OCTA,octa_lowVal,octa_highVal)
                else:
                    #for speckle reduction images or NYU data
                    #frame_3D_resh = np.float32(frame_3D_resh)
                    if self.preview_mode != True:
                        np.save(directory+'/frame_3D.npy',frame_3D_resh)
                    self.image_ready.emit(frame_3D_resh,directory,lowVal,highVal,temp_QI,frame_OCTA,octa_lowVal,octa_highVal)
                self.updateProgressBar.emit(6)
                self.progress.emit(9,0)
                
            # %% Display Image Flythrough
            # fig, (ax1, ax2) = plt.subplots(2, 1 ,gridspec_kw={'height_ratios': [1, 1]})
            # fig.patch.set_alpha(0)
            # ii = 0

            # enface_low = np.min(enface)
            # enface_hight = np.max(enface)
            # enface = ((np.clip((enface - enface_low) / (enface_hight - enface_low), 0, 1)) * 255).astype(np.uint8)

            # bar_f = 2*processParameters.xrng/processParameters.zrng
            # enface_dim = int(bar_f*1024)
            # print("enface_dim:",enface_dim)
            # enface = cv2.resize(enface,(enface_dim,enface_dim),interpolation=cv2.INTER_LINEAR)

            # bscan = frame_3D_resh[:,:,ii]
            # ef = ax1.imshow(np.flipud(np.rot90(enface)), cmap='gray', aspect = 'equal')
            # horizontal_line = ax1.axhline(y=0, color='red', linestyle='-')
            # ax1.axis('off')
            # im = ax2.imshow(bscan, cmap = 'gray', origin = "lower", vmin = lowVal, vmax = highVal)
            # ax2.axis('off')
            # print("bar_f:",bar_f)
            # #ax2.set_aspect(bar_f)
            # current_fig = plt.gcf()
            # current_size = current_fig.get_size_inches()
            # current_fig.set_size_inches(1.6 * current_size[0], 1.6 * current_size[1])
            # dim3 = np.shape(frame_3D_resh)[2]
            # print("dim3:",dim3)
            # def updatefig(frame):
            #     bscan = cv2.resize(np.fliplr(frame_3D_resh[:,1:-1,frame]),(int(1024*bar_f), 1024))
            #     im.set_array(20*np.log10(bscan))
            #     ef.set_array(np.rot90(np.fliplr(np.flipud(enface))))
                
            #     horizontal_line.set_ydata(frame)
            #     return im,


            # ani = animation.FuncAnimation(fig, updatefig, frames = dim3, repeat = False, interval = 50, blit=True)
            # ani.save(directory+'/Bscan_Flythru.gif',writer = 'pillow', fps=90)

            del frame_3D_resh
            #del ani
            self.updateProgressBar.emit(7)
            self.progress.emit(10,0)
        #self.finished.emit()
