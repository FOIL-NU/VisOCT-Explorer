import numpy as np
import GPUtil
import scipy.stats as stats
GPU_available = True
try:
    GPUtil.getGPUs()
    import cupy as cp
    import cupyx.scipy.fft
    from octFuncs import applyBulkAlignment
    from custom_register_images import register_images
except ValueError:
    GPU_available = False
    from image_registration import register_images


def horizontal_flip(frame_3D_resh):
    for ii in range(frame_3D_resh.shape[2]):
        if not ii%2:
            frame_3D_resh[:,:,ii] = np.fliplr(frame_3D_resh[:,:,ii])
    #return frame_3D_resh
            
# Auto-detect Backlash
def get_backlash_pattern(frame_3D_resh,res_slow,vert_shift,progress=None):
    count = 0
    backLashes = np.zeros((int(frame_3D_resh.shape[2]/2),1))
    for ii in range(1,frame_3D_resh.shape[2]):
        refBscan = frame_3D_resh[:,:,ii-1]
        curBscan = frame_3D_resh[:,:,ii]
        shift = register_images(refBscan, curBscan,usfac = 1)
        if ii%2:
            
            backLashes[count] = int(shift[0])
                #print(backLashes)
            count = count + 1
        
        if len(vert_shift) == 0:
            vert_shift.append(int(shift[1]))
        else:
            vert_shift.append(int(vert_shift[-1]+int(shift[1])))
        if progress!= None:
            progress.emit(5)
        
    vert_shift = np.array(vert_shift)
    mid_shift = vert_shift[int(frame_3D_resh.shape[2]/2)]
    vert_shift = vert_shift-mid_shift
    backLash = int(stats.mode(backLashes).mode)
    return backLash

#print("backlash param is:",processParameters.backLash)

def vertical_correction(frame_3D_resh,res_slow,vert_shift,bulk_num=None):
    to_GPU = cp.asarray(frame_3D_resh)
    for ii in range(1,frame_3D_resh.shape[2]):
        if GPU_available:
            refBscan = to_GPU[:,:,ii-1]
            curBscan = to_GPU[:,:,ii]
            shift = register_images(refBscan, curBscan,usfac = 1)
            if len(vert_shift) == 0:
                vert_shift.append(int(shift[1]))
            else:
                vert_shift.append(int(vert_shift[-1] + int(shift[1])))
            #frame_3D_resh[:,:,ii] = np.roll(frame_3D_resh[:,:,ii],-int(shift[1]),0)
        else:
            refBscan = frame_3D_resh[:,:,ii-1]
            curBscan = frame_3D_resh[:,:,ii]
            shift = register_images(refBscan, curBscan,usfac = 1)
            #frame_3D_resh[:,:,ii] = np.roll(curBscan,-int(shift[1]),0)

    del to_GPU


def shift_correction_bulk(frame_3D_resh,bulk_num):
    to_GPU = cp.asarray(frame_3D_resh)
    for ii in range(int(frame_3D_resh.shape[2]/bulk_num)):
        #print(ii)
        frame_stack = to_GPU[:,:,ii*bulk_num:(ii+1)*bulk_num]
        refBscan = frame_stack[:,:,int(bulk_num/2)]
        for frame_num in range(0,bulk_num):
            curBscan = frame_stack[:,:,frame_num]
            shift = register_images(refBscan,curBscan,usfac=1)
            curBscan = cp.roll(curBscan,-int(shift[1]),0)
            frame_stack[:,:,frame_num] = curBscan#cp.roll(curBscan,-int(shift[0]),1)
            del shift
        to_GPU[:,:,ii*bulk_num:(ii+1)*bulk_num] = frame_stack
        del frame_stack
    return to_GPU



# Correct Backlash
def horizontal_correction(frame_3D_resh,res_slow,backLash):

    for ii in range(frame_3D_resh.shape[2]):
        if ii%2:
            frame_3D_resh[:,:,ii] = np.roll(frame_3D_resh[:,:,ii],-backLash,1)

    
        
        
