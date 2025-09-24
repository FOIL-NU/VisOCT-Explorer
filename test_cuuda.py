import numpy as np
import scipy,GPUtil,time
import scipy.interpolate
import scipy.signal.windows
from scipy.fft import fft, ifft, fftshift, fft2, ifft2, ifftshift
from matplotlib import pyplot as plt
import math,os
from ctypes import*

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

'''
dll = CDLL("VS_2022_CUDA_12_2.dll", winmode=0)


init_interp = getattr(dll, "?init_interp@@YAHPEAN00000IPEAD@Z")
init_interp.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint, c_char_p]
init_interp.restype = c_int


        #_declspec (dllexport) int init_interp_bal(double* pixA2B, double* idxABal, double* idxBBal, double* MatABal, const unsigned int resaxis, const char* waveMapA2BFile);
init_interp_bal = getattr(dll, "?init_interp_bal@@YAHPEAN000IPEBD@Z")
init_interp_bal.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint, c_char_p]
init_interp_bal.restype = c_int

        #init_cuda_bal.argtypes = [c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_uint), POINTER(c_uint), POINTER(c_float)]
init_sub_band = getattr(dll, "?init_sub_band@@YAHHHHIPEAI0PEAM11@Z")
init_sub_band.argtypes = [c_int, c_int, c_int, c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
init_sub_band.restype = c_int



dispersion_sub_band_reconstruction_bal = getattr(dll,"?dispersion_sub_band_reconstruction_bal@@YAHPEAM0IMM@Z")
dispersion_sub_band_reconstruction_bal.argtypes = [POINTER(c_float),POINTER(c_float),c_uint,c_float,c_float]
dispersion_sub_band_reconstruction_bal.restype = c_int

dispersion_shift = getattr(dll,"?dispersion_shift@@YAMHIH@Z")
dispersion_shift.argtypes = [c_int, c_uint, c_int]
dispersion_shift.restype = c_float
'''


def calcDispersionPhase(freq_lin,c2a,c3a):
    c2 = c2a*1e-28
    c3 = c3a*1e-40
    
    fdd = freq_lin
    fc = np.mean(freq_lin)
    
    dispersion = c2*(2*math.pi*(fdd-fc))**2 + c3*(2*math.pi*(fdd-fc))**3
    
    return dispersion

# %%
'''
def disperse_cuda(x,testFrm_c,stft_Win_c,res_fast,frame_sub,progress=None):

    c2a_c = c_float(x[0])
    c3a_c = c_float(x[1])
    c_2048 = c_int(2048)
    c_5 = c_int(5)
    res_fast_c = c_uint(res_fast)
    dispersion_out = dispersion_sub_band_reconstruction_bal(frame_sub,testFrm_c,res_fast_c,c2a_c,c3a_c)
    total_shift = dispersion_shift(c_2048,res_fast_c,c_5)
    python_float_value = float(total_shift)
            
    if progress!=None:
        progress.emit(3)
    print("shift:",python_float_value)
    return python_float_value
'''

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
    
    distMat = np.arange(0,1024)
    stepN = 10
    
    dispstftIn.winNum = 5
    dispstftIn.resol = 15e-6
    dispstftIn.firstWave = 530e-9
    dispstftIn.lastWave = 585e-9

    #note that the cuda_dll flag is set to True
    disp_stft = calculateWindows(dispstftIn,wavelength,processParams,processParams.pixelMap,True)
    if GPU_available:
        disp_stft.stft_w_g = cp.asarray(disp_stft.stftWin)
    newres_fast = processParams.res_fast
    if True:
        cc2 = 0
        cc3 = 0
        dispersion = calcDispersionPhase(kmap.freq_lin,cc2,cc3)
            # dispersion_g = cp.asarray(dispersion)

        frameN = round(processParams.res_slow/2)    
            
        testFrm = raw[int(processParams.res_fast*frameN):int(processParams.res_fast*frameN+processParams.res_fast),:]

        print(np.shape(testFrm))
            
        if np.shape(testFrm)[0] > 512:
            testFrm = testFrm[np.arange(0,np.shape(testFrm)[0],int(np.shape(testFrm)[0]/512)),:]
            newres_fast = int(np.shape(testFrm)[0])
                
        print(np.shape(testFrm))
            

    
    
    #print("disperse return value:",type(disperse(x0,testFrm,kmap,disp_stft,processParams,newres_fast,distMat)))
    if GPU_available:
        chunkSz = 16384     
        res_axis = 2048
        res_fast = 512
        band_num = 5

        wavelengthFile = b"wavelength_blizz_06"
        pxMapFile = os.getcwd()+'\\Pixel Maps\\Audi_OCT_Bal_20_34_35_512_512_Rect_X800um_Y800um___SF34_to_SF33_cal.mat'
        pxMapFile = pxMapFile.encode('ascii')
        dll = CDLL("VS_2022_CUDA_12_2.dll", winmode=0)


        init_interp = getattr(dll, "?init_interp@@YAHPEAN00000IPEAD@Z")
        init_interp.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint, c_char_p]
        init_interp.restype = c_int

        zeroVect = [0]*res_axis                             # zero vector
        zeroVect_pt = ((c_double)*len(zeroVect))(*zeroVect)
        res_axis_c = c_uint(res_axis)
        wavelength_c = ((c_double)*len(zeroVect))(*zeroVect)                           # wavelength vector in c
        freq_lin_c = ((c_double)*len(zeroVect))(*zeroVect)                                # linear frequency vector in c
        freq_c = ((c_double)*len(zeroVect))(*zeroVect)                                    # frequency vector in c
        idxA_c = ((c_double)*len(zeroVect))(*zeroVect)                                   # index A vector in c
        idxB_c = ((c_double)*len(zeroVect))(*zeroVect)                                    # index B vector in c
        MatA_c = ((c_double)*len(zeroVect))(*zeroVect)                                    # interpolation Matrix vector in c
        init_interp_out = init_interp(wavelength_c, freq_lin_c, freq_c, idxA_c, idxB_c, MatA_c, res_axis_c, wavelengthFile)
        print(init_interp_out)


        #_declspec (dllexport) int init_interp_bal(double* pixA2B, double* idxABal, double* idxBBal, double* MatABal, const unsigned int resaxis, const char* waveMapA2BFile);
        init_interp_bal = getattr(dll, "?init_interp_bal@@YAHPEAN000IPEBD@Z")
        init_interp_bal.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint, c_char_p]
        init_interp_bal.restype = c_int

        #init_cuda_bal.argtypes = [c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_uint), POINTER(c_uint), POINTER(c_float)]
        init_sub_band = getattr(dll, "?init_sub_band@@YAHHHHIPEAI0PEAM11@Z")
        init_sub_band.argtypes = [c_int, c_int, c_int, c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        init_sub_band.restype = c_int

        zeroVect = [0]*res_axis                             # zero vector
        zeroVect_pt = ((c_double)*len(zeroVect))(*zeroVect)
        res_axis_c = c_uint(res_axis)

        c_1024 = c_int(1024)
        c_512 = c_int(512)
        c_18 = c_int(18)


        pixA2B_c = ((c_double)*len(zeroVect))(*zeroVect)                           # wavelength vector in c
        idxABal_c = ((c_double)*len(zeroVect))(*zeroVect)                                # linear frequency vector in c
        idxBBal_c = ((c_double)*len(zeroVect))(*zeroVect)                                    # frequency vector in c
        MatABal_c = ((c_double)*len(zeroVect))(*zeroVect)                                   # index A vector in c
        init_interp_bal_out = init_interp_bal(pixA2B_c, idxABal_c, idxBBal_c, MatABal_c, res_axis_c, pxMapFile)

        idxA_c = (c_uint * len(idxA_c))(*[int(x) for x in idxA_c])
        idxB_c = (c_uint * len(idxB_c))(*[int(x) for x in idxB_c])
        MatA_c = (c_float * len(MatA_c))(*[float(x) for x in MatA_c])
        freq_c = (c_float * len(freq_c))(*[float(x) for x in freq_c])
        print(init_interp_bal_out)


        res_fast_c = c_uint(res_fast)
        res_axis_c = c_int(res_axis)

        start_time = time.time()
        testFrm_flat = testFrm.flatten()
        testFrm_c = (c_float * len(testFrm_flat))(*[float(x) for x in testFrm_flat])
        stft_Win_flat = disp_stft.stftWin[:,1:].flatten()
        stft_Win_c = (c_float * len(stft_Win_flat))(*[float(x) for x in stft_Win_flat])

        init_sub_band_out = init_sub_band(c_1024,c_512,c_18,res_fast_c,idxA_c,idxB_c,MatA_c,freq_c,stft_Win_c)
        print(init_sub_band_out)

        dispersion_sub_band_reconstruction_bal = getattr(dll,"?dispersion_sub_band_reconstruction_bal@@YAHPEAM0IMM@Z")
        dispersion_sub_band_reconstruction_bal.argtypes = [POINTER(c_float),POINTER(c_float),c_uint,c_float,c_float]
        dispersion_sub_band_reconstruction_bal.restype = c_int

        dispersion_shift = getattr(dll,"?dispersion_shift@@YAMHIH@Z")
        dispersion_shift.argtypes = [c_int, c_uint, c_int]
        dispersion_shift.restype = c_float

        free_cuda_memory = getattr(dll,"?destructor_subpixel_registration@@YAHXZ")
        free_cuda_memory.argtypes = []
        free_cuda_memory.restype = c_int

        zeroVector = [0]*res_fast*res_axis*5
        frame_sub = ((c_float)*len(zeroVector))(*zeroVector)

        def disperse_cuda(x,testFrm_c,stft_Win_c,res_fast,frame_sub,progress=None):

            c2a_c = c_float(x[0])
            c3a_c = c_float(x[1])
            c_2048 = c_int(2048)
            c_5 = c_int(5)
            res_fast_c = c_uint(res_fast)
            print(testFrm_c)
            dispersion_out = dispersion_sub_band_reconstruction_bal(frame_sub,testFrm_c,res_fast_c,c2a_c,c3a_c)
            total_shift = dispersion_shift(c_2048,res_fast_c,c_5)
            python_float_value = float(total_shift)
            if progress!=None:
                progress.emit(3)
            print("shift:",python_float_value)
            return python_float_value


        x0 = [1,0]
        print(testFrm_flat)
        ex = minimize(disperse_cuda, x0, args = (testFrm_c,stft_Win_c,res_fast,frame_sub,progress),method = 'nelder-mead', options = {'xatol': 1e-4,'fatol': 1e-8})
        print(ex.x)
        c2a = float(ex.x[0])
        c3a = float(ex.x[1])
        free_cuda_memory()
        del dll

    else:
        ex = minimize(disperse, x0, args = (testFrm,kmap,disp_stft,processParams,newres_fast,distMat,progress), 
                  method = 'nelder-mead', options = {'xatol': 1e-4,'fatol': 1e-8})
        c2a = ex.x[0]
        c3a = ex.x[1]

    return c2a,c3a
