import os
import tkinter as tk
from tkinter import filedialog
import re,scipy.io
import numpy as np

# Define process parameters class
# %% Parse File Details and Assign File Parameters

class processParams:
    """    
    Process parameters class. This class get initialized with raw fringe file path, pixel map file path and number of chunks

    Class variables:

    :cvar res_axis: (int) number of detector elements on spectrometer. Initial value: 2048
    :cvar upSampleFactor: (int) FFT upsampling factor. Initial value:6
    :cvar zeroPaddingFactor: (int) FFT zero padding factor. Initial value:1
    :cvar zrng: (int) z range in um. Initial value:2387

    Instance variables:

    :ivar chunks: (int) number of chunks sending to balance fringe
    :ivar fname: (str) full path of raw fringe file
    :ivar balFlag: (bool) if the raw data need to be balanced
    :ivar res_fast: (int) number of A-lines along fast axis
    :ivar res_slow: (int) number of A-lines along slow axis
    :ivar xrng: (int) x range in um
    :ivar yrng: (int) y range in um
    :ivar scanMode: (str) Determine scan parameter, 'raster' or 'circ'
    :ivar totalNumberOfAlines: (int) total number of A-lines = res_fast * res_slow
    :ivar backLash: (int) backLash mode of the scan
    :ivar pixelMap: (numpy array) pixel map read from file
    """
    
    res_axis = 2048                                                             # number of detector elements on spectrometer
    #change this for CUDA processing
    upSampleFactor = 6                                                          # FFT upsampling factor
    zeroPaddingFactor = 1                                                       # FFT zero padding factor
    zrng = 2387                                                                 # z range in um

    

    def __init__(self,chunks,file_path,match_path=None):
        """
        Initialize the processParams object

        Parameters
        ----------
        chunks

            (int) number of chunks for processing
        
        file_path

            (str) raw fringe file path

        match_path
            (str) pixel map file path
        """
        self.chunks = chunks
        self.eye = 3
        

        dir0 = os.path.dirname(file_path)                                                       # Extract file directory
        fname = os.path.basename(file_path)                                                    # Extract file name from path
        if match_path!=None:
            fname = fname[0:(len(fname)-4)]

        self.excel_fname = dir0+'/Measurements/'+fname 
        
        self.adaptive_pix_map = fname+'_pixelMap'                                                    # Remove file extension

        fileInfo = fname.split('_')                                                            # parse filename for scan parameters
        self.fname = f"{dir0}{'/'}{fname}{'.RAW'}"
        self.human = False
        if 'OD' in fileInfo:
            self.eye = 0
            self.human = True
        if 'OS' in fileInfo:
            self.eye = 1
            self.human = True
        

        if 'SR' in fileInfo:
            self.SRFlag = 1
            self.newSRFlag = 1 if 'New' in fileInfo else 0
        else:
            self.SRFlag = 0
            self.newSRFlag = 0
        # Determine if file is balanced
        if 'Bal' in fileInfo:
            print(65535)
            self.balFlag = 1
            bal_index = fileInfo.index('Bal')
            self.hr = int(fileInfo[bal_index+1])
            self.min = int(fileInfo[bal_index+2])
            self.sec = int(fileInfo[bal_index+3])
            self.res_fast = int(fileInfo[bal_index+4])                                              # number A-lines along fast axis
            self.res_slow = int(fileInfo[bal_index+5])                                              # number A-lines along slow axis
            self.xrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[bal_index+7])][0]    # x range in um
            self.yrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[bal_index+8])][0]    # y range in um
            if 'Angio' in fileInfo:
                self.octaFlag = 1
                #self.repNum = int(fileInfo[bal_index+11])
                #self.volNum = int(fileInfo[bal_index+13])
                self.res_fast = int(fileInfo[bal_index+4])                                              # number A-lines along fast axis
                self.res_slow = int(fileInfo[bal_index+5])                                              # number A-lines along slow axis
                self.xrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[bal_index+7])][0]    # x range in um
                self.yrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[bal_index+8])][0]    # y range in um
                self.repNum = int(fileInfo[bal_index+10])
                self.volNum = int(fileInfo[bal_index+12])
            else:
                self.repNum = 2
                self.octaFlag = 0

        else:
            self.balFlag = 0
            self.hr = 0
            self.min = 0
            self.sec = 0
            if 'Angio' in fileInfo or 'OCTA' in fileInfo:
                if 'OCTA' in fileInfo:
                    self.balFlag = 0
                else:
                    self.balFlag = 0
                self.octaFlag = 1
                self.repNum = 2                 #OCTA temp
                self.volNum = 5                 #Filename does not contain the information needed
                self.res_fast = 512             #Information hard coded for now
                self.res_slow = 512             #Change in future
                self.xrng = 1000
                self.yrng = 1000
            else:
                print(fileInfo[7])
                self.octaFlag = 0
                self.res_fast = int(fileInfo[5])                                              # number A-lines along fast axis
                self.res_slow = int(fileInfo[6])                                              # number A-lines along slow axis
                self.xrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[8])][0]       # x range in um
                self.yrng = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileInfo[9])][0]       # y range in um
            

        # Determine scan parameter
        if 'Rect' in fileInfo:
            self.scanMode = 'raster'
        elif 'Circ' in fileInfo:
            self.scanMode = 'circ'
            
        if match_path != None:
            self.match_path = match_path.encode('ascii')
            if match_path == "within":
                with open(self.fname.split('.RAW')[0][0:(len(self.fname)-5)]+'1.RAW','rb') as f:
                    f.seek(-16 * 1024, 2)  # 2 means seek from the end
                    # Read the last 16 KB
                    b = f.read()

                    self.pixelMap = np.frombuffer(b).copy()
                    f.close()
                    print("OK")
                    print(self.repNum)
                    print(self.volNum)
            elif match_path.split('.')[-1] != 'mat':
                if match_path != "None":
                    with open(match_path, 'rb') as f:
                        b = f.read()

                    self.pixelMap = np.frombuffer(b).copy()
                    f.close()
                else:
                    temp = np.arange(0,2048)
                    self.pixelMap = temp
            else:
                self.pixelMap = np.asarray(scipy.io.loadmat(match_path).get('hold_max')[0])

        self.totalNumberOfAlines = (self.res_fast*self.res_slow*self.repNum) if self.octaFlag else (self.res_fast*self.res_slow)
        if self.newSRFlag:
            self.totalBnum = self.res_slow
            self.numAvgs = int(os.stat(self.fname).st_size/(self.res_fast*self.res_slow*2*self.res_axis))
            self.res_slow = self.numAvgs*self.res_slow
            self.totalNumberOfAlines = int(os.stat(self.fname).st_size/(2*self.res_axis))
        self.backLash = 0
        self.dispersion = 0
        print("res fast:",self.res_fast)
        if self.yrng == 0:
            self.yrng = 1