import numpy as np
import GPUtil,time,gc
from PySide6.QtCore import Signal
from scipy.signal.windows import tukey
from scipy.optimize import minimize
from PySide6.QtCore import QThread
#import cupyx.scipy.interpolate
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    from cupyx.scipy import interpolate
except ValueError:
    GPU_available = False

import scipy.interpolate

class adaptive_balance:
    """calculating the balanced fringes by subtracting the auto-correlation term of both arm from sample arm"""
    def __init__(self, fpath, processParams):
        self.fpath = fpath
        print(fpath)
        fpath1 = fpath.split('.RAW')[0][0:(len(fpath)-5)]+'1.RAW'
        fpath2 = fpath.split('.RAW')[0][0:(len(fpath)-5)]+'2.RAW'
        batch_size = 2*4096*2048
        with open(fpath1, 'rb') as f:
            file1 = f.read(batch_size)
        f.close()
        with open(fpath2, 'rb') as f:
            file2 = f.read(batch_size)
        f.close()
        self.processParams = processParams
        self.chunkSz = 1*4096*2048

        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('=')
    # Open and read binary file
        alinesPerChunk = int(self.chunkSz/processParams.res_axis)                                # number of A-lines per chunk
        
        raw1 = np.frombuffer(file1,dtype = dt, count = int(self.chunkSz), offset = 0)     # read pixel values from loaded buffer
        self.adaptive_raw1 = np.reshape(raw1,(alinesPerChunk,int(processParams.res_axis)))                  # resahpe to normal OCT view

        raw2 = np.frombuffer(file2,dtype = dt, count = int(self.chunkSz), offset = 0)     # read pixel values from loaded buffer
        adaptive_raw2 = np.reshape(raw2,(alinesPerChunk,int(processParams.res_axis)))                  # resahpe to normal OCT view

        self.adaptive_background2 = cp.mean(adaptive_raw2,axis=0)
        self.adpative_subtractor = cp.asarray(adaptive_raw2/self.adaptive_background2)

    
    
    def get_adaptive_balance_Fringes(self,interpMat,progress = None):
        """
        Return
        ----------
            Numpy array. Balanced fringes of the whole volume by concatenating all the fringe numpy array chunks into one numpy array
        """
        processParams = self.processParams
        chunkSz = 2048*processParams.res_axis
        offsetAmt = 0
        
        if GPU_available:
            raw1Interp = cp.zeros(cp.shape(self.adaptive_raw1))

            raw1Interp = interpolate.pchip_interpolate(range(cp.shape(self.adaptive_raw1)[1]),self.adaptive_raw1,interpMat,axis=1)

            background1 = cp.mean(raw1Interp, axis=0)

            raw = raw1Interp/background1 - self.adpative_subtractor

            raw[:,interpMat<=0] = 0
            raw[:,interpMat>=2047] = 0
            #del raw1
            #del raw2
            #del interpMat
            #gc.collect()
            del raw1Interp
            del background1
                #cp.get_default_memory_pool().free_all_blocks()
                #gc.collect()
            return raw
        else:
            data = self.__balanceFringes(offsetAmt,chunkSz)

        return data
    


class Adaptive_Balance_Thread(QThread):
    finished = Signal()
    optimized = Signal(np.ndarray,str)

    def __init__(self, fpath, processParameters, ui):
        self.processParameters = processParameters
        self.raw = adaptive_balance(fpath, processParameters)
        print("class created")
        self.ui = ui
        super().__init__()

    def run(self):     
        ex1 = minimize(self.adaptive_balance, 1, args = (self.raw,self.processParameters), 
                  method = 'nelder-mead', options = {'xatol': 1e-3,'fatol': 1e-8})

        ex2 = minimize(self.adaptive_balance, [1, ex1.x[0]], args = (self.raw,self.processParameters), 
                        method = 'nelder-mead', options = {'xatol': 1e-3,'fatol': 1e-8})

        ex3 = minimize(self.adaptive_balance, [0, ex2.x[0], ex2.x[1]], args = (self.raw,self.processParameters), 
                        method = 'nelder-mead', options = {'xatol': 1e-3,'fatol': 1e-8})
        
        self.ui.textEdit_2.append("Adaptive balance completed.")
        del self.raw
        self.optimized.emit(ex3.x,self.processParameters.adaptive_pix_map)

    def adaptive_balance(self,coefficients,raw,processParameters):
        res_axis = 2048
        pixel_array = cp.linspace(1,res_axis,res_axis)
        if len(coefficients)<=2:
            evalRng = np.arange(0,25)
        else:
            evalRng = np.arange(0,80)
        
        print(coefficients)
        
        if len(coefficients)==1:
            pixMap = cp.polyval(cp.asarray([1,coefficients[0]]), pixel_array)
        else:
            pixMap = cp.polyval(cp.asarray(coefficients), pixel_array)
            
        pixMap[pixMap<0] = 0
        
        pixMap[pixMap>res_axis] = res_axis
        
        first_zeros = int(cp.count_nonzero(pixMap==0)+5)
        last_zeros = int(cp.count_nonzero(pixMap==res_axis)+5)
        tukeyArray = np.zeros(res_axis)
        alpha = 0.25
        tukeyArray[first_zeros:res_axis-last_zeros] = tukey(res_axis-first_zeros-last_zeros,alpha)
        #print(tukeyArray)
        balanced_raw = raw.get_adaptive_balance_Fringes(pixMap)
        
        bscan = abs(cp.fft.fft(balanced_raw*cp.asarray(tukeyArray)))
        meanAline = cp.mean(bscan,0)
        meanAline = 20*cp.log10(meanAline[0:int(res_axis/2)])

        
        pixDiff = pixel_array - pixMap
        #print(coefficients)
        sumPixDiff = 0
        if coefficients[-1] > 0:
            sumPixDiff = abs(cp.sum(pixDiff[pixDiff>0]))
        else:
            sumPixDiff = abs(cp.sum(pixDiff[pixDiff<0]))
            
        print("sumPixDiff:",sumPixDiff)
        pixPenalty = cp.count_nonzero(pixMap==0) + cp.count_nonzero(pixMap==(res_axis)) + sumPixDiff
        print("pixPenalty:",pixPenalty)
        
        noiseVar = float(cp.var(bscan[0:80]) + cp.sum(meanAline[evalRng]) + pixPenalty)
        
        self.ui.textEdit_2.append("Minimizing noise variance:"+str(noiseVar))
        self.ui.textEdit_2.verticalScrollBar().setValue(self.ui.textEdit_2.verticalScrollBar().maximum())
        return noiseVar