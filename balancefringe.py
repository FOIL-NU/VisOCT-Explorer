import numpy as np
import GPUtil,time,gc
from PySide6.QtCore import Signal
from matplotlib import pyplot as plt
#import cupyx.scipy.interpolate
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    #from scipy import interpolate
    #from univariatespline import InterpolatedUnivariateSpline
    from cupyx.scipy import interpolate
    from cupyx.scipy.signal import hilbert
except ValueError:
    GPU_available = False

import scipy.interpolate

class fringes:
    """calculating the balanced fringes by subtracting the auto-correlation term of both arm from sample arm"""
    def __init__(self, fpath, processParams, pixelMap, seek_no = 0):
        self.fpath = fpath
        fpath1 = fpath.split('.RAW')[0][0:(len(fpath)-5)]+'1.RAW'
        fpath2 = fpath.split('.RAW')[0][0:(len(fpath)-5)]+'2.RAW'
        if processParams.octaFlag:
            #if processParams.res_fast==1024:
            #    batch_size = 5 * 1024 * 1024 * 1024
            #else:
            batch_size = processParams.repNum * processParams.res_fast * processParams.res_slow * processParams.res_axis * 2
        
        self.interpMat = pixelMap
        #print(pixelMap)
        self.processParams = processParams
        self.chunkSz = processParams.totalNumberOfAlines*processParams.res_axis/processParams.chunks
        #self.chunkSz = processParams.totalNumberOfAlines*processParams.res_axis
        if processParams.balFlag:
            with open(fpath1, 'rb') as f:
                if processParams.octaFlag:
                    f.seek(seek_no*batch_size)
                    self.file1 = f.read(batch_size)
                else:
                    self.file1 = f.read()
            f.close()
            try:
                with open(fpath2, 'rb') as f:
                    if processParams.octaFlag:
                        f.seek(seek_no*batch_size)
                        self.file2 = f.read(batch_size)
                    else:
                        self.file2 = f.read()
                f.close()
            except (IOError,FileNotFoundError):
                print("2nd spectrometer file does not exist")
            self.adaptive_raw1 = self.__loadRawFringes(self.file1,0,2*4096*processParams.res_axis)
            adaptive_raw2 = self.__loadRawFringes(self.file2,0,2*4096*processParams.res_axis)
            self.adaptive_background2 = cp.mean(adaptive_raw2,axis=0)
            self.adpative_subtractor = cp.asarray(adaptive_raw2/self.adaptive_background2)
        else:
            with open(fpath, 'rb') as f:
                if processParams.octaFlag:
                    f.seek(seek_no*batch_size)
                    self.file1 = f.read(batch_size)
                else:
                    self.file1 = f.read()
        f.close()
        


    def __loadRawFringes(self, b, offsetSz,chunkSz):
        """
        Private method. Return raw fringes numpy array

        Parameters
        ----------
        fpath

            (str) providing the path of the raw fringe file

        offsetSz

            (int) indicating the offset amount of current chunk

        Return
        ----------
            Numpy array. Raw fringes
        """
        processParams = self.processParams
    # Establish data type for RAW file
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('=')
    
    # Open and read binary file
    
        alinesPerChunk = int(chunkSz/processParams.res_axis)                                # number of A-lines per chunk
        
        raw = np.frombuffer(b,dtype = dt, count = int(chunkSz), offset = int(offsetSz))     # read pixel values from loaded buffer
        raw = np.reshape(raw,(alinesPerChunk,int(processParams.res_axis)))                  # resahpe to normal OCT view
        return raw


    def __balanceFringes_gpu(self,offsetAmt,chunkSz,interpMat=None):
        interpMat = self.interpMat
        raw1 = cp.array(self.__loadRawFringes(self.file1,offsetAmt,chunkSz))
        raw2 = cp.array(self.__loadRawFringes(self.file2,offsetAmt,chunkSz))

        startime = time.time()

        raw1Inter = interpolate.pchip_interpolate(cp.arange(1,cp.shape(raw1)[1]+1),raw1,interpMat,axis=1)

        background1 = cp.mean(raw1Inter, axis=0)
        background2 = cp.mean(raw2,axis=0)
        ratio = background2/background1

        raw = raw1Inter*ratio - raw2

        raw[:,interpMat<=0] = 0
        raw[:,interpMat>=2047] = 0
        print(self.processParams.envelop)
        if self.processParams.envelop:

            envelop = cp.zeros_like(raw)
            for i in range(cp.shape(raw)[0]):
                aline = raw[i,:]
                envelop[i,:] = cp.abs(hilbert(aline))
            envelop_avg = cp.mean(envelop,axis=0)
            raw = raw/envelop_avg
            del envelop

        endtime = time.time()
        print(round(endtime-startime,2))
        del raw1
        del raw2
        del interpMat
        gc.collect()
        return raw


    def __balanceFringes(self,offsetAmt,chunkSz):
        """
        Private method.

        Parameters
        ----------
        offsetAmt

            (int) indicating offset amount of the current chunk

        Return
        ----------
            Numpy array. Current chunk of balanced fringes
        """
        interpMat = self.interpMat
        #interpMat[interpMat > 2047] = 2047
        raw1 = self.__loadRawFringes(self.file1,offsetAmt,chunkSz)
        raw2 = self.__loadRawFringes(self.file2,offsetAmt,chunkSz)

        raw1Interp = np.zeros(np.shape(raw1))

        raw1Interp = scipy.interpolate.pchip_interpolate(range(np.shape(raw1)[1]),raw1,interpMat,axis=1)

        background1 = np.mean(raw1Interp, axis=0)
        background2 = np.mean(raw2,axis=0)

        raw = raw1Interp/background1 - raw2/background2
        print("raw shape:",np.shape(raw))
        raw[:,interpMat<=0] = 0
        raw[:,interpMat>=2047] = 0
        print("raw min",np.min(raw))

        del raw1
        del raw2
        return raw
        

    def get_balance_Fringes(self,interpMat=None,progress = None):
        """
        Return
        ----------
            Numpy array. Balanced fringes of the whole volume by concatenating all the fringe numpy array chunks into one numpy array
        """
        processParams = self.processParams
        chunkSz = self.chunkSz
        data = np.zeros((processParams.totalNumberOfAlines,processParams.res_axis))

        for curChunk in range(processParams.chunks):
            if progress!=None:
                progress.emit(1)
            offsetAmt = int((curChunk)*chunkSz*2)
            startInd = int(offsetAmt/(2*processParams.res_axis))
            endInd = startInd + int((chunkSz/processParams.res_axis))
        
            if GPU_available:
                if interpMat is None:
                    temp = self.__balanceFringes_gpu(offsetAmt,chunkSz)
                    #return temp
                else:
                    temp = self.__balanceFringes_gpu(offsetAmt,chunkSz,interpMat)
                
                data[startInd:endInd,:] = cp.asnumpy(temp)
                del temp
                #cp.get_default_memory_pool().free_all_blocks()
                #gc.collect()
            else:
                data[startInd:endInd,:] = self.__balanceFringes(offsetAmt)

        return data
    
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

    def get_adaptive_balance_Fringes_result(self,interpMat,progress = None):
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

            raw = raw*cp.asarray(self.adaptive_background2)

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


    def get_unbalance_Fringes(self,progress = None):
        """
        Return
        ----------
            Numpy array. Unbalanced fringes (raw fringes) of the whole volume
        """
        fpath = self.fpath
        processParams = self.processParams
        chunkSz = processParams.totalNumberOfAlines*processParams.res_axis
        offsetAmt = 0
        return self.__DirectLoadFringes(fpath, chunkSz, offsetAmt, processParams)
    
    def __DirectLoadFringes(self,fpath,chunkSz, offsetSz, processParams):
    
        # Establish data type for RAW file
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('=')
        
        # Open and read binary file
        with open(fpath, 'rb') as f:
            b = f.read()
        
        alinesPerChunk = int(chunkSz/processParams.res_axis)                                # number of A-lines per chunk
        
        # print(np.asarray([offsetSz,offsetSz+chunkSz]))
        
        raw = np.frombuffer(b,dtype = dt, count = int(chunkSz), offset = int(offsetSz))     # read pixel values from loaded buffer
        raw = np.reshape(raw,(alinesPerChunk,int(processParams.res_axis)))                  # resahpe to normal OCT view
        #print("raw shape is:",raw.shape)
        raw = raw - np.mean(raw,axis=0)
        return raw


