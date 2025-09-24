import numpy as np
import scipy,GPUtil
import scipy.interpolate
import scipy.signal.windows
from scipy.fft import fft, ifft, fftshift, fft2, ifft2, ifftshift
from matplotlib import pyplot as plt
import math
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    import cupyx.scipy.fft
    from custom_register_images import register_images
except ValueError:
    GPU_available = False
    from image_registration import register_images

import numpy as np
from matplotlib import pyplot as plt
from octFuncs import calculateWindows,octRecon_Bal_GPU,octRecon_Bal
# import cupyx.scipy.interpolate
from scipy.optimize import minimize

def calcDispersionPhase(freq_lin,c2a,c3a):
    c2 = c2a*1e-28
    c3 = c3a*1e-40
    
    fdd = freq_lin
    fc = np.mean(freq_lin)
    
    dispersion = c2*(2*math.pi*(fdd-fc))**2 + c3*(2*math.pi*(fdd-fc))**3
    
    return dispersion

# %%
def disperse_GPU(x,testFrm,kmap,disp_stft,processParams,newres_fast,distMat,progress=None):
    cc2 = x[0]
    cc3 = x[1]
    
    usfac = 18
    
    dispersion = calcDispersionPhase(kmap.freq_lin,cc2,cc3)
    dispersion_g = cp.asarray(dispersion)
    shift = 0
    
    frameW = octRecon_Bal_GPU(testFrm,dispersion_g,disp_stft.stft_w_g,processParams,newres_fast,kmap)
    frameW = cp.asarray(np.abs(frameW))

    refWindow = np.ceil((np.shape(disp_stft.stft_w_g)[1]-1)/2)
    #refWindow = 2

    
    for wind in range(1,np.shape(disp_stft.stft_w_g)[1]):
        output = register_images(frameW[distMat,:,int(refWindow)],frameW[distMat,:,int(wind)],usfac=usfac)
        shift = shift + abs(output[1])
    
    #frameWfull = frameW[distMat,:,0]
    if progress!=None:
        progress.emit(3)
    print("total shift:", float(shift))
    del frameW
    return float(shift)


def disperse(x,testFrm,kmap,disp_stft,processParams,newres_fast,distMat,progress=None):
    cc2 = x[0]
    cc3 = x[1]
    
    usfac = 18
    
    dispersion = calcDispersionPhase(kmap.freq_lin,cc2,cc3)
    shift = 0
    
    frameW = octRecon_Bal(testFrm,dispersion,disp_stft.stftWin,processParams,newres_fast,kmap)
    frameW = np.abs(frameW)

    refWindow = np.ceil((np.shape(disp_stft.stftWin)[1]-1)/2)
    #refWindow = 2

    
    for wind in range(1,np.shape(disp_stft.stftWin)[1]):
        output = register_images(frameW[distMat,:,int(refWindow)],frameW[distMat,:,int(wind)],usfac=usfac)
        shift = shift + abs(output[1])
    
    frameWfull = frameW[distMat,:,0]
    if progress!=None:
        progress.emit(3)
    del frameW
    return float(shift)

# %%
def dispersionOptimization_balance(raw,wavelength,processParams,interpMat,kmap, frmGiven,progress=None):
    
    class dispstftIn:
        pass
    
    distMat = np.arange(100,900)
    stepN = 10
    
    dispstftIn.winNum = 5
    dispstftIn.resol = 15e-6
    dispstftIn.firstWave = 530e-9
    dispstftIn.lastWave = 585e-9
    disp_stft = calculateWindows(dispstftIn,wavelength,processParams,interpMat)
    if GPU_available:
        disp_stft.stft_w_g = cp.asarray(disp_stft.stftWin)
    newres_fast = processParams.res_fast
    if not frmGiven:
        cc2 = 0
        cc3 = 0
        dispersion = calcDispersionPhase(kmap.freq_lin,cc2,cc3)
        # dispersion_g = cp.asarray(dispersion)
        Imax = 0
        frameN = 0

        for ii in range(0,processParams.res_slow,stepN):
            #print("calc")
            rawTemp = raw[int(processParams.res_fast*ii):int(processParams.res_fast*ii+processParams.res_fast),:]
            if GPU_available:
                alines_to_GPU = 4096
                tempF = octRecon_Bal_GPU(rawTemp,cp.asarray(dispersion),disp_stft.stft_w_g[:,0],processParams,alines_to_GPU,kmap)
            else:
                tempF = octRecon_Bal(rawTemp,dispersion,disp_stft.stftWin[:,0],processParams,processParams.res_fast,kmap)

            tempF = np.squeeze(tempF)
            
            tempF = abs(tempF[distMat,:])
            temp = np.sort(tempF,0)
            temp = np.mean(temp[100:150,:],0)
            temp = np.sort(temp)
            temp = np.mean(temp[1:10])
            
            if (temp>Imax):
            #if ii==10:
                Imax = temp
                frameN = ii
                img = tempF
        #plt.imshow(np.flipud(img),cmap = 'gray', vmin = lowVal, vmax = highVal)
        #plt.show()
        
        if frameN == 0:
            frameN = round(processParams.res_slow*0.5)
        print(frameN)
        #testFrm = np.flipud(raw[int(processParams.res_fast*frameN):int(processParams.res_fast*frameN+processParams.res_fast),:])
        testFrm = raw[int(processParams.res_fast*frameN):int(processParams.res_fast*frameN+processParams.res_fast),:]

        print(np.shape(testFrm))
        
        if np.shape(testFrm)[0] > 512:
            testFrm = testFrm[np.arange(0,np.shape(testFrm)[0],int(np.shape(testFrm)[0]/512)),:]
            newres_fast = int(np.shape(testFrm)[0])
            
        print(np.shape(testFrm))
    else:
        testFrm = raw
        if np.shape(testFrm)[0] > 512:
            testFrm = testFrm[np.arange(0,np.shape(testFrm)[0],int(np.shape(testFrm)[0]/512)),:]
            newres_fast = int(np.shape(testFrm)[0])
        

    
    x0 = np.asarray([1, 0])
    
    #print("disperse return value:",type(disperse(x0,testFrm,kmap,disp_stft,processParams,newres_fast,distMat)))
    if GPU_available:
        ex = minimize(disperse_GPU, x0, args = (testFrm,kmap,disp_stft,processParams,newres_fast,distMat,progress), 
                  method = 'nelder-mead', options = {'xatol': 1e-4,'fatol': 1e-8})
        c2a = ex.x[0]
        c3a = ex.x[1]
        cp.get_default_memory_pool().free_all_blocks()
    else:
        ex = minimize(disperse, x0, args = (testFrm,kmap,disp_stft,processParams,newres_fast,distMat,progress), 
                  method = 'nelder-mead', options = {'xatol': 1e-4,'fatol': 1e-8})
        c2a = ex.x[0]
        c3a = ex.x[1]
    print("c2a:",c2a)
    print("c3a:",c3a)
    return c2a,c3a