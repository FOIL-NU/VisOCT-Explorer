import numpy as np
import matplotlib.pyplot as plt
# from scipy import interpolate
# import cupy as cp
import os
import h5py
from ctypes import*
import time

start_time = time.time()
chunkSz = 16384     
res_axis = 2048
res_fast = 512
res_slow = 16384//res_fast

wavelengthFile = b"./Wavelength Files/wavelength_blizz_06"    # wavelength file location
pxMapFile = b"pixelMap_Audi_OCT_Bal_17_37_41_512_512_Rect_X2000um_Y2000um___NU18_to_NU17_cal"

dll = CDLL("visOCT_balance.dll",winmode=0)

#_declspec (dllexport) int init_interp(double* wavelength, double* freq_lin, double* freq, double* idxA, double* idxB, double* MatA, const unsigned int resaxis, char* wavelengthFile);
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
init_interp.restype = c_int

zeroVect = [0]*res_axis                             # zero vector
zeroVect_pt = ((c_double)*len(zeroVect))(*zeroVect)
res_axis_c = c_uint(res_axis)
pixA2B_c = ((c_double)*len(zeroVect))(*zeroVect)                           # wavelength vector in c
idxABal_c = ((c_double)*len(zeroVect))(*zeroVect)                                # linear frequency vector in c
idxBBal_c = ((c_double)*len(zeroVect))(*zeroVect)                                    # frequency vector in c
MatABal_c = ((c_double)*len(zeroVect))(*zeroVect)                                   # index A vector in c
init_interp_bal_out = init_interp_bal(pixA2B_c, idxABal_c, idxBBal_c, MatABal_c, res_axis_c, pxMapFile)
print(init_interp_bal_out)

# _declspec (dllexport) int init_cuda_bal(const unsigned int res_tot, const unsigned int* idxA, const unsigned int* idxB, const float* MatA, const float* freq, const unsigned int* idxABal, const unsigned int* idxBBal, const float* MatABal);
init_cuda_bal = getattr(dll, "?init_cuda_bal@@YAHIPEBI0PEBM1001@Z")
init_cuda_bal.argtypes = [c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_uint), POINTER(c_uint), POINTER(c_float)]
init_cuda_bal.restype = c_int

res_tot =res_fast * res_slow
res_tot_c = c_uint(res_tot)
idxA_c = (c_uint * len(idxA_c))(*[int(x) for x in idxA_c])
idxB_c = (c_uint * len(idxB_c))(*[int(x) for x in idxB_c])
MatA_c = (c_float * len(MatA_c))(*[float(x) for x in MatA_c])
freq_c = (c_float * len(freq_c))(*[float(x) for x in freq_c])
idxABal_c = (c_uint * len(idxABal_c))(*[int(x) for x in idxABal_c])
idxBBal_c = (c_uint * len(idxBBal_c))(*[int(x) for x in idxBBal_c])
MatABal_c = (c_float * len(MatABal_c))(*[float(x) for x in MatABal_c])
init_cuda_bal_out = init_cuda_bal(res_tot_c, idxA_c, idxB_c, MatA_c, freq_c, idxABal_c, idxBBal_c, MatABal_c)
print(init_cuda_bal_out)

# import raw data
# input_file = 'D:\\OCT Data\\0212 HD\\OCT_16_12_01__VIS1.RAW'
# npimg = np.fromfile(input_file, dtype=np.uint16)
# raw1_all = np.reshape(npimg, (-1, 2048),'A')
# input_file = 'D:\\OCT Data\\0212 HD\\OCT_16_12_01__VIS1.RAW'
# npimg = np.fromfile(input_file, dtype=np.uint16)
# raw2_all = np.reshape(npimg,(-1, 2048),'A')
# testdt = h5py.File("D:/qt test/231212 HD/16_57_31_OCT.h5", 'r')

# testdt1 = "C:/OCT Data/NoID/Mar_21_2025/Audi_OCT_Bal_10_56_34_512_512_Rect_X1900um_Y2000um__1.raw"
# with open(testdt1, 'rb') as f:
#     raw1_all = f.read()
# dt = np.dtype(np.uint16)
# dt = dt.newbyteorder('=')


# raw1_all = np.frombuffer(raw1_all,dtype = dt, count = int(16384*4*2048), offset = 0)     # read pixel values from loaded buffer
# raw1_all = np.reshape(raw1_all,(16384*4,res_axis))   
testdt1 = np.load('my_array_1.npy')
raw1_all = np.reshape(testdt1,(16384,res_axis))

testdt2 = np.load('my_array_2.npy')
# with open(testdt2, 'rb') as f:
#     raw2_all = f.read()
    
# raw2_all = np.frombuffer(raw2_all,dtype = dt, count = int(16384*4*2048), offset = 0)     # read pixel values from loaded buffer
# raw2_all = np.reshape(raw2_all,(16384*4,res_axis))   

raw2_all = np.reshape(testdt2,(16384,res_axis))
noA = 512
noB = 32
# plt.plot(raw1_all[1000,:])
# plt.plot(raw2_all[1000,:])
# plt.show()
chunkNum = 1
# chunkNum = 2
print(chunkNum)
frm1_all = np.zeros((res_fast, res_axis,chunkNum), dtype = np.single)
frm2_all = np.zeros((res_fast, res_axis,chunkNum), dtype = np.single)
frm3_all = np.zeros((res_fast, res_axis,chunkNum), dtype = np.single)
enface_all = np.zeros((res_slow*chunkNum, res_fast), dtype = np.single)

start_time2 = time.time()

for icc in range(chunkNum):
    print(icc)
    raw1 = raw1_all[chunkSz*icc:chunkSz*(icc+1),:]
    raw2 = raw2_all[chunkSz*icc:chunkSz*(icc+1),:]
    print(raw1.shape, res_slow, res_fast)

    # testdt = h5py.File('230707 HD/11_30_32_OCT.h5', 'r')


    # print(testdt.keys())
    # raw1 = (np.squeeze(np.asarray(testdt['dataset_1']).astype(np.uint16)))
    # raw2 = (np.squeeze(np.asarray(testdt['dataset_2']).astype(np.uint16)))

    # _declspec (dllexport) int preview_bal(float* background1, float* background2, const unsigned __int16* raw1, const unsigned __int16* raw2, const unsigned int res_tot, const float c2a, const float c3a);
    preview_bal = getattr(dll, "?preview_bal@@YAHPEAM0PEBG1IMM@Z")
    ptr_2d_uint16 = np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2)
    preview_bal.argtypes = [POINTER(c_float), POINTER(c_float), ptr_2d_uint16, ptr_2d_uint16, c_uint, c_float, c_float]
    preview_bal.restype = c_int

    background1_c = ((c_float)*len(zeroVect))(*zeroVect) 
    background2_c = ((c_float)*len(zeroVect))(*zeroVect) 
    res_tot_c = c_uint(res_tot)
    c2a_c  = c_float(-0.01)
    c3a_c = c_float(-.001)
    preview_bal_out = preview_bal(background1_c, background2_c, raw1, raw2, res_tot_c, c2a_c, c3a_c)
    

    extractThreeBscanWithCuda = getattr(dll,"?extractThreeBscanWithCuda@@YAHPEAMI0I0IIIH@Z")
    #_declspec (dllexport) int extractThreeBscanWithCuda(float* frm1, const unsigned int loc1, float* frm2, const unsigned int loc2, float* frm3, const unsigned int loc3, const unsigned int res_x, const unsigned int res_y, const int BS);

    ptr_2d_uint16 = np.ctypeslib.ndpointer(dtype=np.single, ndim=2)
    extractThreeBscanWithCuda.argtypes = [ptr_2d_uint16, c_uint, ptr_2d_uint16, c_uint, ptr_2d_uint16, c_uint, c_uint, c_uint, c_int]
    extractThreeBscanWithCuda.restype = c_int
    # frm1 = np.zeros((res_fast, res_axis), dtype = np.single)
    # frm2 = np.zeros((res_fast, res_axis), dtype = np.single)
    # frm3 = np.zeros((res_fast, res_axis), dtype = np.single)
    loc1 = c_uint(0)
    loc2 = c_uint(16)
    loc3 = c_uint(20)
    res_x = c_uint(res_fast)
    res_y = c_uint(res_slow)
    bs = c_int(24)

    extractThreeBscanWithCuda_out = extractThreeBscanWithCuda(frm1_all[:,:,icc], loc1, frm2_all[:,:,icc], loc2, frm3_all[:,:,icc], loc3, res_x, res_y, bs)
    print(frm1_all)
    # plt.subplots(1,3)
    # plt.subplot(131); plt.imshow(np.log10(frm1), 'gray')
    # plt.subplot(132); plt.imshow(np.log10(frm2), 'gray')
    # plt.subplot(133); plt.imshow(np.log10(frm3), 'gray')
    # plt.show()

    #_declspec (dllexport) int projectWithCuda(float* frame_2D, const unsigned int res_tot, const unsigned int lower, const unsigned int upper);
    projectWithCuda = getattr(dll, '?projectWithCuda@@YAHPEAMIII@Z')

    ptr_2d_float = np.ctypeslib.ndpointer(dtype=np.single, ndim=2)
    projectWithCuda.argtypes = [ptr_2d_float, c_uint, c_uint, c_uint]
    projectWithCuda.restype = c_int

    frame_2D = np.zeros((res_slow, res_fast), dtype = np.single)
    lower_c = c_uint(16)
    upper_c = c_uint(800)
    projectWithCuda_out = projectWithCuda(frame_2D, res_tot_c, lower_c, upper_c)

    #_declspec (dllexport) int flip2DWithCuda(float* frame_flip, const float* frame, const unsigned int res_x, const unsigned int res_y, const int BS);
    flip2DWithCuda = getattr(dll, '?flip2DWithCuda@@YAHPEAMPEBMIIH@Z')

    flip2DWithCuda.argtypes = [ptr_2d_float, ptr_2d_float, c_uint, c_uint, c_int]
    flip2DWithCuda.restype = c_int

    frame_flip = np.zeros((res_slow, res_fast), dtype = np.single)
    frame = frame_2D
    flip2DWithCuda_out = flip2DWithCuda(frame_flip, frame, res_x, res_y, bs)

    #_declspec (dllexport) int substractWithCuda(float* frm_o, const float* frm_i, const float num, const unsigned int element_tot);
    substractWithCuda = getattr(dll, '?substractWithCuda@@YAHPEAMPEBMMI@Z')
    substractWithCuda.argtypes = [ptr_2d_float, ptr_2d_float, c_float, c_uint]
    substractWithCuda.restype = c_int

    frm_o_sub = np.zeros((res_slow, res_fast), dtype = np.single)
    frm_i = np.log(frame_flip)
    num_c = c_float(0)
    substractWithCuda_out = substractWithCuda(frm_o_sub, frm_i, num_c, res_tot_c)

    # _declspec (dllexport) int divideWithCuda(float* frm_o, const float* frm_i, const float num, const unsigned int element_tot);
    divideWithCuda = getattr(dll, '?divideWithCuda@@YAHPEAMPEBMMI@Z')
    divideWithCuda.argtypes = [ptr_2d_float, ptr_2d_float, c_float, c_uint]
    divideWithCuda.restype = c_int

    frm_o_div = np.zeros((res_slow, res_fast), dtype = np.single)
    frm_i = frm_o_sub
    num_c = 1
    divideWithCuda_out = divideWithCuda(frm_o_div, frm_i, num_c, res_tot_c)

    enface_all[res_slow*icc:res_slow*(icc+1),:] = frm_o_div
    del frm_o_div
enface_all = np.transpose(enface_all)
enface_all = np.rot90(enface_all)
enface_all = np.fliplr(enface_all)

# plt.imshow(frm1_all[:,:,0])
# plt.show()
# plt.imshow(frm2_all[:,:,-1])
# plt.show()

print('time all:', time.time()-start_time)
print('time prev:', time.time()-start_time2)
plt.imshow(np.rot90(np.fliplr(np.log(frm2_all[20:,:1024,0]))), 'gray', aspect = 1,vmin=7, vmax=10)


plt.show()

