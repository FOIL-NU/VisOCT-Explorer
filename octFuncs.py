import numpy as np
import scipy
import time
import GPUtil
#import torch
import scipy.interpolate
import scipy.signal.windows
from scipy.fft import fft, ifft, fftshift, fft2, ifft2, ifftshift
#from skimage.registration import phase_cross_correlation
#from custom_phase_cross_correlation import phase_cross_correlation_gpu
#from image_registration import register_images
from matplotlib import pyplot as plt
import math
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    import cupyx.scipy.fft
    import cupyx.scipy.signal
    from custom_register_images import register_images
except ValueError:
    GPU_available = False
import numpy as np
from matplotlib import pyplot as plt
# import cupyx.scipy.interpolate
from scipy.optimize import minimize

# %%
# %%
def normpdf(x, mu, sigma):
    
    y = (1/(sigma*math.sqrt(2*math.pi))) * np.exp(-(x-mu)**2 / (2*sigma**2))
    return y

# %%
def calculateWindows(stft,wavelength,processParams,interpMat,cuda_dll=False):
    cc = 2.99792458e8
    c1 = 2*math.log(2)/math.pi
    freq = cc / (wavelength)
    
    f1 = scipy.interpolate.interp1d(range(processParams.res_axis),freq)
    freq = f1(np.linspace(0,processParams.res_axis-1,processParams.res_axis*processParams.upSampleFactor))
    if cuda_dll:
        freq = f1(np.linspace(0,processParams.res_axis-1,processParams.res_axis))
    
    freq_max = freq[0]
    freq_min = freq[-1]
    freq_lin = np.linspace(freq_min,freq_max,len(freq))
    freq_lin = np.flipud(freq_lin)    
    
    stft.stftWin = np.zeros((len(freq_lin),stft.winNum+1))
    
    firstZeros = sum(interpMat==0)+5
    lastZeros = sum(interpMat==2048)+5
    
    tukeyWindow = scipy.signal.windows.tukey(processParams.res_axis-firstZeros-lastZeros,0.25)
    tukeyWindow = np.append(np.zeros((firstZeros,1)), tukeyWindow)
    tukeyWindow = np.append(tukeyWindow,np.zeros((lastZeros,1)))
    
    f11 = scipy.interpolate.interp1d(range(processParams.res_axis),tukeyWindow)
    tukeyWindow = f11(np.linspace(0,processParams.res_axis-1,processParams.res_axis*processParams.upSampleFactor))
    if cuda_dll:
        tukeyWindow = f11(np.linspace(0,processParams.res_axis-1,processParams.res_axis))

    stft.stftWin[:,0] = tukeyWindow
    
    firstK = cc/stft.firstWave
    lastK = cc/stft.lastWave
    
    kCents = np.linspace(firstK,lastK,stft.winNum)
    lambCents = cc/kCents
    lambWidths = c1*(lambCents)**2/stft.resol
    
    f2 = scipy.interpolate.interp1d(range(processParams.res_axis),wavelength)
    waveLong = f2(np.linspace(0,processParams.res_axis-1,processParams.res_axis*processParams.upSampleFactor))
    if cuda_dll:
        waveLong = f2(np.linspace(0,processParams.res_axis-1,processParams.res_axis))

    centInds = np.zeros((stft.winNum,1))
    for pp in range(stft.winNum):
        centInds[pp] = np.argmin(abs(lambCents[pp]-waveLong))
        lamb1 = lambCents[pp]-lambWidths[pp]/2
        lamb2 = lambCents[pp]+lambWidths[pp]/2
        pix1 = np.argmin(abs(lamb1-waveLong))
        pix2 = np.argmin(abs(lamb2-waveLong))
        sigma = abs((pix2-pix1)/2*math.sqrt(2*math.log(2)))
        winTemp = normpdf(np.asarray(range(len(waveLong))),centInds[pp],sigma)
        winTemp = winTemp/max(winTemp)
        
        f3 = scipy.interpolate.interp1d(waveLong,winTemp)
        winFin = f3(cc/freq_lin)
        stft.stftWin[:,pp+1] = winFin
    
    return stft

# %%
def octRecon_Bal(raw,dispersion,stft,processParams,alinesToGPU,kmap):

    try:
        numBands = np.shape(stft)[1]
    except:
        numBands = 1
    frame_3D = np.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    
    startInds = np.arange(0,np.shape(raw)[0],alinesToGPU)
    endInds = np.zeros((np.shape(startInds)[0],1))

    for jj in range(0,np.shape(raw)[0],alinesToGPU):
        if jj + alinesToGPU > np.shape(raw)[0]:
            endChunk = np.shape(raw)[0]
        else:
            endChunk = jj + alinesToGPU

    for jj in range(len(startInds)):
        if startInds[jj] + alinesToGPU > np.shape(raw)[0]:
            endInds[jj] = np.shape(raw)[0]
        else:
            endInds[jj] = startInds[jj] + alinesToGPU
    
    finDepth = int(processParams.res_axis/2)
    
    
    for ii in range(len(startInds)):
        startIndex = int(startInds[ii])
        endIndex = int(endInds[ii])

        fringe_gpu = (raw[startIndex:endIndex,:])
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:-1] = 0
        
        fringe_fft_gpu = fftshift(fft(fringe_gpu))
        del fringe_gpu
        
        fringe_fft_gpu[:,np.shape(raw)[1]-8:np.shape(raw)[1]+8] = 0
        pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)
        fringe_fft_gpu = fftshift(np.pad(fringe_fft_gpu,((0,0),(pdAmt, pdAmt))))
        fringe_fft_gpu = np.real(ifft(fringe_fft_gpu))
        
        print(np.shape(fringe_fft_gpu))
        
        print(np.shape(fringe_fft_gpu))
        fringe_lin =np.squeeze(kmap.MatA*fringe_fft_gpu[:,kmap.idxA.astype(int)] + kmap.MatB*fringe_fft_gpu[:,kmap.idxB.astype(int)])
        del fringe_fft_gpu
        
        for bandNum in range(0,numBands):
            if numBands == 1:
                fringe_lin_win = stft * fringe_lin
            else:
                fringe_lin_win = stft[:,bandNum] * fringe_lin
            
            dispersionPhaseTerm = np.exp(-1j * dispersion)
            fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
            frame = np.fft.fft(fringe_lin_win)
                
            frame = frame[:,0:int(processParams.res_axis/2)]
            
            frame_3D[startIndex:endIndex,:,bandNum] = abs((frame))

        
    frame_3D_out = frame_3D
    frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
        
    return frame_3D_out

def octangio_Recon_Bal_GPU_STFT_window(raw,dispersion,stft_w_g,processParams,kmap):
    res_fast = processParams.res_fast
    repNum = processParams.repNum

    frame_3D = np.zeros((int(processParams.res_axis/2), res_fast, processParams.res_slow))
    #frame_OCTA = np.zeros((int(processParams.res_axis/2), res_fast,  processParams.res_slow))
    print("raw dim is:",np.shape(raw))
    #bands = [0,5,6,7,8,9,10]

    pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)

    startIndex = range(0,np.shape(raw)[0],res_fast*repNum)
    print("shape of startIndex is",np.shape(startIndex))
    finDepth = int(np.shape(raw)[0]/2)
    for i in range(len(startIndex)):
        #temp3D = cp.zeros((int(processParams.res_axis/2),res_fast,6*(repNum-1)))
        temp = cp.zeros((512,1024,2),dtype = cp.complex128)
        if startIndex[i]+res_fast*repNum > np.shape(raw)[0]:
            endIndex = np.shape(raw)[0]
        else:
            endIndex = startIndex[i] + res_fast*repNum
        
        fringe_gpu = cp.asarray(raw[startIndex[i]:endIndex,:])
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:-1] = 0

        fringe_fft_gpu = cp.fft.fft(fringe_gpu)
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)

        del fringe_gpu
            
        fringe_fft_gpu[:,finDepth-8:finDepth+8] = 0
        fringe_fft_gpu = cp.pad(fringe_fft_gpu,((0,0),(pdAmt, pdAmt)))

        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
                
        fringe_fft_gpu = cp.real(cp.fft.ifft(fringe_fft_gpu))

        fringe_lin = cp.squeeze(kmap.MatA_g*fringe_fft_gpu[:,kmap.idxA_g.astype(int)] + kmap.MatB_g*fringe_fft_gpu[:,kmap.idxB_g.astype(int)])

        counter = 0
                #print("shape of fringe_lin is", cp.shape(fringe_lin))
                #print("shape of stft_w_g is:", cp.shape(stft_w_g))
        fringe_lin_win = stft_w_g * fringe_lin
        dispersionPhaseTerm = cp.exp(-1j * dispersion)
        fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
        zeroPadAmt = int((processParams.zeroPaddingFactor - 1)*int(cp.shape(fringe_lin_win)[0])/2)
        fringe_lin_win = cp.pad(fringe_lin_win,((0,0),(zeroPadAmt,zeroPadAmt)))

        frame = cp.fft.fft(fringe_lin_win,axis=1)
        frame = frame[:,0:int(cp.shape(frame)[1]/(2*processParams.upSampleFactor))]
        for j in range(0,repNum):
            temp[:,:,j] = frame[j*int((cp.shape(frame)[0])/2):(j+1)*int((cp.shape(frame)[0])/2),:]
        temp1 = cp.transpose(temp,(1,0,2))
            #if band_id == 0:
                    #print("shape of fringe_lin_win is",cp.shape(fringe_lin_win))
        temp1 = cp.abs(temp1)
        h_shift = cp.zeros((repNum,1))
        v_shift = cp.zeros((repNum,1))
        for dd in range(1,repNum):
            frame1 = temp1[:,:,0]
            frame2 = temp1[:,:,dd]
            shift = register_images(frame1, frame2, usfac = 1)
            h_shift[dd] = shift[0]
            v_shift[dd] = shift[1]
            temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(v_shift[dd])),1)
            temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(h_shift[dd])),0)
        fullFrame = cp.squeeze(cp.mean(temp1,axis=2))
        frame_3D[:,:,i] = fullFrame.get()
        #frame_3D[:,:,i] = temp1[:,:,0].get()
               
    frame_3D_out = frame_3D
    #frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
    return frame_3D_out

def octRecon_Bal_GPU_STFT_window(raw,dispersion,stft_w_g,processParams,alinesToGPU,kmap):

    try:
        numBands = cp.shape(stft_w_g)[1]
    except:
        numBands = 1
    
    #frame_3D = cp.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    frame_3D_out = np.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    
    pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)   
    #print("pad amount",pdAmt)  
    
    startInds = np.arange(0,np.shape(raw)[0],alinesToGPU)
    endInds = np.zeros((np.shape(startInds)[0],1))
    
    for jj in range(len(startInds)):
        if startInds[jj] + alinesToGPU > np.shape(raw)[0]:
            endInds[jj] = np.shape(raw)[0]
        else:
            endInds[jj] = startInds[jj] + alinesToGPU
    
    finDepth = int(processParams.res_axis/2)
    count = 0
    for ii in range(len(startInds)):
        count += 1
        startIndex = int(startInds[ii])
        endIndex = int(endInds[ii])
        
        fringe_gpu = cp.asarray(raw[startIndex:endIndex,:])
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:-1] = 0
        
        #with plan1:
        startTime = time.time()
        fringe_fft_gpu = cp.fft.fft(fringe_gpu)
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
        
        del fringe_gpu
        
        fringe_fft_gpu[:,finDepth-8:finDepth+8] = 0
        fringe_fft_gpu = cp.pad(fringe_fft_gpu,((0,0),(pdAmt, pdAmt)))
        
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
        
        #with plan2:              
        fringe_fft_gpu = cp.real(cp.fft.ifft(fringe_fft_gpu))
        
        #print("fringe min",cp.min(fringe_fft_gpu))
        fringe_lin = cp.squeeze(kmap.MatA_g*fringe_fft_gpu[:,kmap.idxA_g.astype(int)].copy() + kmap.MatB_g*fringe_fft_gpu[:,kmap.idxB_g.astype(int)].copy())

        del fringe_fft_gpu
        endTime = time.time()
        #print("cupy FFT time",endTime-startTime)
        
        #for bandNum in range(0,numBands):
        fringe_lin_win = stft_w_g * fringe_lin
            
        dispersionPhaseTerm = cp.exp(-1j * dispersion)
        fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
            #with plan3:
        frame = cp.fft.fft(fringe_lin_win)
                
        frame = frame[:,0:int(processParams.res_axis/2)].copy()
            #print(frame[10,:])
        frame_3D_out[startIndex:endIndex,:,0] = abs(cp.asnumpy((frame)))
        del frame
        del fringe_lin
        del fringe_lin_win
        

    frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
    
    return frame_3D_out

def octangio_Recon_Bal_GPU(raw,dispersion,stft_w_g,processParams,kmap):
    res_fast = processParams.res_fast
    repNum = processParams.repNum
    #repNum = 2

    frame_3D = np.zeros((int(processParams.res_axis/2), res_fast, processParams.res_slow))
    frame_OCTA = np.zeros((int(processParams.res_axis/2), res_fast,  processParams.res_slow))
    print("raw dim is:",np.shape(raw))
    bands = [0,5,6,7,8,9,10]

    pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)

    startIndex = range(0,np.shape(raw)[0],res_fast*repNum)
    print("shape of startIndex is",np.shape(startIndex))
    finDepth = int(np.shape(raw)[1]/2)
    for i in range(len(startIndex)):
        temp3D = cp.zeros((int(processParams.res_axis/2),res_fast,6*(repNum-1)))
        temp = cp.zeros((res_fast,1024,repNum),dtype = cp.complex128)
        if startIndex[i]+res_fast*repNum > np.shape(raw)[0]:
            endIndex = np.shape(raw)[0]
        else:
            endIndex = startIndex[i] + res_fast*repNum
        
        fringe_gpu = cp.asarray(raw[startIndex[i]:endIndex,:])
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:-1] = 0

        fringe_fft_gpu = cp.fft.fft(fringe_gpu)
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)

        del fringe_gpu
            
        fringe_fft_gpu[:,finDepth-8:finDepth+8] = 0
        fringe_fft_gpu = cp.pad(fringe_fft_gpu,((0,0),(pdAmt, pdAmt)))

        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
                
        fringe_fft_gpu = cp.real(cp.fft.ifft(fringe_fft_gpu))

        fringe_lin = cp.squeeze(kmap.MatA_g*fringe_fft_gpu[:,kmap.idxA_g.astype(int)] + kmap.MatB_g*fringe_fft_gpu[:,kmap.idxB_g.astype(int)])

        counter = 0
        for band_id in bands:
                #print("shape of fringe_lin is", cp.shape(fringe_lin))
                #print("shape of stft_w_g is:", cp.shape(stft_w_g))
            fringe_lin_win = stft_w_g[:,band_id] * fringe_lin
            dispersionPhaseTerm = cp.exp(-1j * dispersion)
            fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
            zeroPadAmt = int((processParams.zeroPaddingFactor - 1)*int(cp.shape(fringe_lin_win)[0])/2)
            fringe_lin_win = cp.pad(fringe_lin_win,((0,0),(zeroPadAmt,zeroPadAmt)))

            frame = cp.fft.fft(fringe_lin_win,axis=1)
            frame = frame[:,0:int(cp.shape(frame)[1]/(2*processParams.upSampleFactor))]
            for j in range(0,repNum):
                temp[:,:,j] = frame[j*int((cp.shape(frame)[0])/repNum):(j+1)*int((cp.shape(frame)[0])/repNum),:]
            temp1 = cp.transpose(temp,(1,0,2))
            if band_id == 0:
                    #print("shape of fringe_lin_win is",cp.shape(fringe_lin_win))
                temp1 = cp.abs(temp1)
                #h_shift = cp.zeros((repNum,1))
                #v_shift = cp.zeros((repNum,1))
                for dd in range(1,repNum):
                    frame1 = temp1[:,:,dd-1]
                    frame2 = temp1[:,:,dd]
                    shift = register_images(frame1, frame2, usfac = 1)
                    temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(shift[1])),0)
                    temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(shift[0])),1)
                fullFrame = cp.squeeze(cp.mean(temp1,axis=2))
                #fullFrame = temp1[:,:,0]
                frame_3D[:,:,i] = fullFrame.get()
                
            else:
                frame_s1 = temp1[:,:,0]
                
                for ind in range(1,repNum):
                    #if ind <= 1:
                    frame_s2 = temp1[:,:,ind]
                    frame_temp = frame_s2

                    AGPF = -cp.angle(cp.sum(frame_s2[:,:] * cp.conj(frame_s1[:,:]),axis=0))
                    AGPF2 = cp.tile(AGPF,(int(cp.shape(frame_s1)[0]),1))
                        
                    frame_s2 = cp.reshape(cp.exp(1j*AGPF2),(1024,res_fast),'A') * frame_s2

                    LGPF = -cp.angle(cp.sum(frame_s2*cp.conj(frame_s1),axis=1))
                    LGPF2 = cp.transpose(cp.tile(LGPF,(int(cp.shape(frame_s1)[1]),1)))

                    frame_s2 = cp.exp(1j*LGPF2) * frame_s2 
                    frame_sd = frame_s1 - frame_s2
                    frame_sd = cp.where(frame_sd == 0, 1e-5, frame_sd)
                            
                    temp3D[:,:,counter] = abs(frame_sd)
                        #temp3D[:,:,counter] = cp.angle(frame_s1 * cp.conj(frame_s2)) #np.angle(frame_s2)
                    counter += 1
                    frame_s1 = frame_temp
        tempvar = cupyx.scipy.signal.medfilt2d(cp.squeeze(cp.mean(temp3D,axis=2)))
        #tempvar = temp3D[:,:,0]
        frame_OCTA[:,:,i] = abs(tempvar.get())
        cp.get_default_memory_pool().free_all_blocks()
        
    frame_3D_out = frame_3D
    #frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
    return frame_3D_out,frame_OCTA

# %%
def octRecon_Bal_GPU(raw,dispersion,stft_w_g,processParams,alinesToGPU,kmap):

    try:
        numBands = cp.shape(stft_w_g)[1]
    except:
        numBands = 1
    
    #frame_3D = cp.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    frame_3D_out = np.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    #a1 = cp.zeros((processParams.res_axis,alinesToGPU))
    #plan1 = get_fft_plan(a1,axes=0,value_type='C2C')
    
    #a2 = cp.zeros((processParams.res_axis*processParams.upSampleFactor,alinesToGPU), 'cfloat')
    #plan2 = get_fft_plan(a2,axes=0,value_type='C2C')
    
    #a3 = cp.zeros((processParams.res_axis*processParams.upSampleFactor,alinesToGPU), 'cfloat')
    #plan3 = get_fft_plan(a3,axes=0,value_type='C2C')
        
    pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)   
    #print("pad amount",pdAmt)  
    
    startInds = np.arange(0,np.shape(raw)[0],alinesToGPU)
    endInds = np.zeros((np.shape(startInds)[0],1))
    
    for jj in range(len(startInds)):
        if startInds[jj] + alinesToGPU > np.shape(raw)[0]:
            endInds[jj] = np.shape(raw)[0]
        else:
            endInds[jj] = startInds[jj] + alinesToGPU
    
    finDepth = int(processParams.res_axis/2)
    count = 0
    for ii in range(len(startInds)):
        count += 1
        startIndex = int(startInds[ii])
        endIndex = int(endInds[ii])
        
        fringe_gpu = cp.asarray(raw[startIndex:endIndex,:])
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:] = 0
        
        #with plan1:
        startTime = time.time()
        fringe_fft_gpu = cp.fft.fft(fringe_gpu)
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
        
        del fringe_gpu
        
        fringe_fft_gpu[:,finDepth-8:finDepth+8] = 0
        fringe_fft_gpu = cp.pad(fringe_fft_gpu,((0,0),(pdAmt, pdAmt)))
        
        fringe_fft_gpu = cupyx.scipy.fft.fftshift(fringe_fft_gpu,1)
        
        #with plan2:              
        fringe_fft_gpu = cp.real(cp.fft.ifft(fringe_fft_gpu))
        
        #print("fringe min",cp.min(fringe_fft_gpu))
        fringe_lin = cp.squeeze(kmap.MatA_g*fringe_fft_gpu[:,kmap.idxA_g.astype(int)].copy() + kmap.MatB_g*fringe_fft_gpu[:,kmap.idxB_g.astype(int)].copy())

        del fringe_fft_gpu
        endTime = time.time()
        #print("cupy FFT time",endTime-startTime)
        
        for bandNum in range(0,numBands):
            if numBands == 1:
                fringe_lin_win = stft_w_g * fringe_lin
            else:
                fringe_lin_win = stft_w_g[:,bandNum] * fringe_lin
            
            dispersionPhaseTerm = cp.exp(-1j * dispersion)
            fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
            #with plan3:
            frame = cp.fft.fft(fringe_lin_win)
                
            frame = frame[:,0:int(processParams.res_axis/2)]
            #print(frame[10,:])
            frame_3D_out[startIndex:endIndex,:,bandNum] = abs(cp.asnumpy((frame)))
        del frame
        del fringe_lin
        del fringe_lin_win
        

    frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
    
    return frame_3D_out



'''def octRecon_Bal_Torch(raw,dispersion,stft_w_g,processParams,alinesToGPU,kmap):
    
    try:
        numBands = cp.shape(stft_w_g)[1]
    except:
        numBands = 1

    #raw = torch.as_tensor(raw, device='cuda')
    stft_w_g = torch.as_tensor(stft_w_g, device='cuda')
    dispersion = torch.as_tensor(dispersion,device='cuda')
    #kmap = torch.as_tensor(kmap, device='cuda')

    frame_3D = torch.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    frame_3D_out = np.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
    
    #a1 = cp.zeros((processParams.res_axis,alinesToGPU))
    #plan1 = get_fft_plan(a1,axes=0,value_type='C2C')
    
    #a2 = cp.zeros((processParams.res_axis*processParams.upSampleFactor,alinesToGPU), 'cfloat')
    #plan2 = get_fft_plan(a2,axes=0,value_type='C2C')
    
    #a3 = cp.zeros((processParams.res_axis*processParams.upSampleFactor,alinesToGPU), 'cfloat')
    #plan3 = get_fft_plan(a3,axes=0,value_type='C2C')
        
    pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)     
    
    startInds = np.arange(0,np.shape(raw)[0],alinesToGPU)
    endInds = np.zeros((np.shape(startInds)[0],1))
    
    for jj in range(len(startInds)):
        if startInds[jj] + alinesToGPU > np.shape(raw)[0]:
            endInds[jj] = np.shape(raw)[0]
        else:
            endInds[jj] = startInds[jj] + alinesToGPU
    
    finDepth = int(processParams.res_axis/2)
    count = 0
    for ii in range(len(startInds)):
        count += 1
        startIndex = int(startInds[ii])
        endIndex = int(endInds[ii])
        
        fringe_gpu = torch.as_tensor(raw[startIndex:endIndex,:], device='cuda')
        fringe_gpu[:,0:4] = 0
        fringe_gpu[:,-5:-1] = 0
        
        #with plan1:
        
        startTime = time.time()
        fringe_fft_gpu = torch.fft.fft(fringe_gpu)
        
        fringe_fft_gpu = torch.fft.fftshift(fringe_fft_gpu,1)
        del fringe_gpu

        fringe_fft_gpu[:,finDepth-8:finDepth+8] = 0
        fringe_fft_gpu = torch.nn.functional.pad(fringe_fft_gpu,((pdAmt, pdAmt)))
        
        fringe_fft_gpu = torch.fft.fftshift(fringe_fft_gpu,1)
        
        #with plan2:  
        fringe_fft_gpu = torch.fft.ifft(fringe_fft_gpu)
            
        fringe_fft_gpu = torch.real(fringe_fft_gpu)

        fringe_lin = torch.squeeze(kmap.MatA_t*fringe_fft_gpu[:,kmap.idxA.astype(int)] + kmap.MatB_t*fringe_fft_gpu[:,kmap.idxB.astype(int)])
        del fringe_fft_gpu

        endTime = time.time()
        print("Torch FFT time",endTime-startTime)
        
        for bandNum in range(0,numBands):
            if numBands == 1:
                fringe_lin_win = stft_w_g * fringe_lin
            else:
                fringe_lin_win = stft_w_g[:,bandNum] * fringe_lin
            
            dispersionPhaseTerm = torch.exp(-1j * dispersion)
            fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
            
            #with plan3:
            frame = torch.fft.fft(fringe_lin_win)
                
            frame = frame[:,0:int(processParams.res_axis/2)]
            
            frame_3D[startIndex:endIndex,:,bandNum] = torch.abs((frame))
        
    frame_3D_out = frame_3D.numpy()
    frame_3D_out = np.transpose(frame_3D_out,(1,0,2))
    return frame_3D_out'''

# %%
def applyBulkAlignment(img1,cum_z,cum_y):
    
    #temp = cp.zeros(cp.shape(img1),dtype=cp.complex64)
    img1 = cp.roll(cp.roll(img1,-cum_z,0),-cum_y,1)
    if cum_z > 0:
        img1[-cum_z:,:] = 0
    elif cum_z < 0:
        img1[0:-cum_z,:] = 0
    if cum_y > 0:
        img1[:,-cum_y:] = 0
    elif cum_y < 0:
        img1[:,0:-cum_y] = 0
    '''
    blimit = int(-cum_z)
    ulimit = int(cp.shape(img1)[1] - cum_z)
    rlimit = int(-cum_y)
    llimit = int(cp.shape(img1)[0] - cum_y)
    if blimit<0 and rlimit<0:
        temp[cp.shape(img1)[0]-llimit:cp.shape(img1)[0],cp.shape(img1)[1]-ulimit:cp.shape(img1)[1]] = img1[0:llimit,0:ulimit]
    elif blimit<0 and rlimit>=0:
        temp[0:cp.shape(img1)[0]-rlimit,cp.shape(img1)[1]-ulimit:cp.shape(img1)[1]] = img1[rlimit:cp.shape(img1)[0],0:ulimit]
    elif blimit>=0 and rlimit<0:
        temp[cp.shape(img1)[0]-llimit:cp.shape(img1)[0], 0:cp.shape(img1)[1]-blimit] = img1[0:llimit,blimit:cp.shape(img1)[1]]
    else:
        temp[0:cp.shape(img1)[0]-rlimit,0:cp.shape(img1)[1]-blimit] = img1[rlimit:cp.shape(img1)[0],blimit:cp.shape(img1)[1]]
    '''
        
    return img1

def applyBulkAlignment_np(img1,cum_z,cum_y):
    
    #temp = cp.zeros(cp.shape(img1),dtype=cp.complex64)
    img1 = np.roll(np.roll(img1,-cum_z,0),-cum_y,1)
    if cum_z > 0:
        img1[-cum_z:,:] = 0
    elif cum_z < 0:
        img1[0:-cum_z,:] = 0
    if cum_y > 0:
        img1[:,-cum_y:] = 0
    elif cum_y < 0:
        img1[:,0:-cum_y] = 0
    '''
    blimit = int(-cum_z)
    ulimit = int(cp.shape(img1)[1] - cum_z)
    rlimit = int(-cum_y)
    llimit = int(cp.shape(img1)[0] - cum_y)
    if blimit<0 and rlimit<0:
        temp[cp.shape(img1)[0]-llimit:cp.shape(img1)[0],cp.shape(img1)[1]-ulimit:cp.shape(img1)[1]] = img1[0:llimit,0:ulimit]
    elif blimit<0 and rlimit>=0:
        temp[0:cp.shape(img1)[0]-rlimit,cp.shape(img1)[1]-ulimit:cp.shape(img1)[1]] = img1[rlimit:cp.shape(img1)[0],0:ulimit]
    elif blimit>=0 and rlimit<0:
        temp[cp.shape(img1)[0]-llimit:cp.shape(img1)[0], 0:cp.shape(img1)[1]-blimit] = img1[0:llimit,blimit:cp.shape(img1)[1]]
    else:
        temp[0:cp.shape(img1)[0]-rlimit,0:cp.shape(img1)[1]-blimit] = img1[rlimit:cp.shape(img1)[0],blimit:cp.shape(img1)[1]]
    '''
        
    return img1