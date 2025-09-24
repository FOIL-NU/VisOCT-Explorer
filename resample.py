import numpy as np
import scipy
import GPUtil
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    from cupyx.scipy import interpolate
except ValueError:
    GPU_available = False

class linear_k:
    """    
    linear wavenumber(k) matrix class

    Instance variables:

    :ivar kmap: (kmap) consists of 4 numpy array MatA, MatB, idxA, idxB. idx corresponds to the x-axis and Mat corresponds to y-axis
    """

    def __init__(self):
        """
        Initalize linear_k instance

        Parameters
        ----------
            None

        Return
        ----------
            None
        """
        class kmap:
            pass
        
        self.kmap = kmap

    def calculate_linear_interpolation_matrix(self,wavelength,processParams):
        """
        Convert interferogram linear in wavelength to linear in wavenumber

        Parameters
        ----------
            wavelength
                (numpy array) wavelength from file

            processParams
                (processParams) processing parameters

        Return
        ----------
            None
        """
        kmap = self.kmap
            
        cc = 2.99792458e8
        freq = cc / (wavelength)
            
        f = scipy.interpolate.interp1d(range(processParams.res_axis),freq)
            
        freq = f(np.linspace(0,processParams.res_axis-1,processParams.res_axis*processParams.upSampleFactor))
            
        freq_max = freq[0]
        freq_min = freq[-1]
        kmap.freq_lin = np.linspace(freq_min,freq_max,len(freq))
        kmap.freq_lin = np.flipud(kmap.freq_lin)
            
        j_rt = 1
        kmap.idxA = np.zeros((len(kmap.freq_lin),1))
        kmap.idxB = np.zeros((len(kmap.freq_lin),1))
        kmap.MatA = np.zeros((len(kmap.freq_lin),1))
        kmap.MatB = np.zeros((len(kmap.freq_lin),1))
            
        kmap.idxA[0] = 0
        kmap.idxA[-1] = len(kmap.freq_lin)-1
        kmap.idxB[0] = 0
        kmap.idxB[-1] = len(kmap.freq_lin)-1
        kmap.MatA[0] = 0.5
        kmap.MatA[-1] = 0.5
        kmap.MatB[0] = 0.5
        kmap.MatB[-1] = 0.5
                
        for ind in range((len(kmap.freq_lin))):
                
            while (kmap.freq_lin[ind] < freq[j_rt]) and (j_rt < len(freq)): 
                j_rt = j_rt + 1
                    
            j_lt = j_rt - 1

            a = freq[j_lt] - kmap.freq_lin[ind]
            b = kmap.freq_lin[ind]- freq[j_rt]
                
            kmap.idxA[ind] = j_lt
            kmap.idxB[ind] = j_rt
            kmap.MatA[ind] = b / (a+b)
            kmap.MatB[ind] = a / (a+b)
                

    def To_GPU(self):
        """
        Convert linear_k_matrix numpy array into cupy array
        """
        kmap = self.kmap
        kmap.MatA_g = cp.asarray(kmap.MatA)
        kmap.MatB_g = cp.asarray(kmap.MatB)
        kmap.idxA_g = cp.asarray(kmap.idxA)
        kmap.idxB_g = cp.asarray(kmap.idxB)



#kmap = calculate_linear_interpolation_matrix(data.wavelength,processParameters)


