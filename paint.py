import sys

import psutil
import pyqtgraph as pg

from balancefringe import fringes
from processParams import processParams
from resample import linear_k
from skimage import exposure,morphology
from dispersion import *
import matplotlib.animation as animation
import registration

#import h5py

import cv2
import time,os
import sys
import numpy as np
import scipy.io
import cupyx.scipy.signal

from PIL import Image,ImageEnhance,ImageOps,ImageQt
from octFuncs import *
from random import randint
from sidebar_ui import Ui_MainWindow
#%load_ext memory_profiler
match_Path = 'C:\\Users\\xufen\\Downloads\\_OCT_Bal_20_05_13_512_512_Rect_X3000um_Y3000um_OD_SO2__NU18_to_NU17_cal 1.mat'
f = "C:\\Users\\xufen\\Downloads\\OCT_Bal_13_19_27_512_512_Rect_X3000um_Y3000um_OS_FOV_SO2_1.RAW"
#f = "C:\\Users\\xufen\\Documents\\RaymondA_OCT_Bal_13_01_41_256_256_Rect_X3000um_Y3000um_Rep_2_Vol_0__Angio_1.RAW"
processParameters = processParams(32,f, match_Path)

import matplotlib.pyplot as plt
from PIL import Image


def aline_reg(aline_stack, aline):
    ret = cp.zeros((np.shape(aline_stack)))
    
    aline1 = cp.fft.fft(aline)
    shift = cp.zeros(np.shape(aline_stack)[1])
    for i in range(1,np.shape(aline_stack)[1]):
        aline2 = cp.fft.fft(aline_stack[:,i])
        crs_cor = cp.abs(cp.fft.ifft(aline1*cp.conj(aline2)))
            
        shift[i] = cp.argmax(crs_cor)
        if shift[i] >= 512:
            shift[i] = shift[i] -1024
        
        
    for i in range(cp.shape(shift)[0]):
        ret[:,i] = cp.roll(aline_stack[:,i],int(shift[i]))
        
    return ret


def interactive_red_painter_ms_paint(
    image,
    brush_radius=0,
    show_coords=False,
):
    """
    MS Paint–style interactive red pixel painter using Matplotlib.

    Press & hold left mouse button to paint while dragging.
    Undo entire last stroke with right-click or 'u'/'z'/Backspace.
    Press 'q' or Esc to quit (window closes; function returns modified image array).

    Parameters
    ----------
    image : str | np.ndarray | PIL.Image.Image
        Image path, array, or PIL Image.
        Arrays: (H,W), (H,W,3), or (H,W,4).
    brush_radius : int, default=0
        0 = single pixel; 1 = 3x3; k = (2k+1)^2 square brush.
    show_coords : bool, default=False
        Print stroke stats to console.

    Returns
    -------
    np.ndarray
        Modified RGB uint8 image.
    """

    # ---- Load & normalize to RGB uint8 array ----
    if isinstance(image, str):
        img = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        arr = np.array(image)
        if arr.ndim == 2:  # grayscale -> RGB
            img = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[2] == 4:  # drop alpha
                img = arr[..., :3]
            elif arr.shape[2] == 3:
                img = arr
            else:
                raise ValueError("Unsupported channel count.")
        else:
            raise ValueError("Unsupported image shape.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    img = img.copy()
    nrows, ncols = img.shape[:2]

    all_painted_points = []

    # ---- Undo stack: list of (coords_list, old_vals_array) ----
    action_stack = []

    # ---- Stroke state ----
    painting = False
    stroke_coords = []      # list of (r,c) for this stroke (unique)
    stroke_old_vals = []    # list of [R,G,B] old pixel values
    stroke_seen = set()     # to avoid duplicates within a stroke
    last_r = None
    last_c = None

    # ---- Helpers ----
    def in_bounds(r, c):
        return 0 <= r < nrows and 0 <= c < ncols

    def get_brush_pixels(r, c, radius):
        r0, r1 = max(0, r - radius), min(nrows - 1, r + radius)
        c0, c1 = max(0, c - radius), min(ncols - 1, c + radius)
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        return rr.ravel(), cc.ravel()

    def apply_brush_at(r, c):
        if not in_bounds(r, c):
            return
        rr, cc = get_brush_pixels(r, c, brush_radius)
        for R, C in zip(rr, cc):
            key = (int(R), int(C))
            if key not in stroke_seen:
                stroke_seen.add(key)
                stroke_coords.append(key)
                stroke_old_vals.append(img[R, C].copy())
                all_painted_points.append(key)  # <-- record globally
        img[rr, cc] = [255, 0, 0]

    def line_pixels(r0, c0, r1, c1):
        """Return integer-sampled pixels along a line (inclusive)."""
        length = max(abs(r1 - r0), abs(c1 - c0)) + 1
        rr = np.linspace(r0, r1, length).round().astype(int)
        cc = np.linspace(c0, c1, length).round().astype(int)
        return rr, cc

    def start_stroke(r, c):
        nonlocal painting, last_r, last_c, stroke_coords, stroke_old_vals, stroke_seen
        painting = True
        last_r, last_c = r, c
        stroke_coords = []
        stroke_old_vals = []
        stroke_seen = set()
        apply_brush_at(r, c)

    def continue_stroke(r, c):
        nonlocal last_r, last_c
        if last_r is None or last_c is None:
            start_stroke(r, c)
            return
        # interpolate so we don't miss gaps
        rr, cc = line_pixels(last_r, last_c, r, c)
        for R, C in zip(rr, cc):
            apply_brush_at(int(R), int(C))
        last_r, last_c = r, c

    def end_stroke():
        nonlocal painting, stroke_coords, stroke_old_vals
        if not painting:
            return
        painting = False
        if stroke_coords:
            # push to undo stack
            coords = stroke_coords[:]  # list of (r,c)
            old_vals = np.array(stroke_old_vals, dtype=np.uint8)
            action_stack.append((coords, old_vals))
            if show_coords:
                print(f"Stroke painted: {len(coords)} px. Undo depth={len(action_stack)}")

    def undo_last():
        if not action_stack:
            if show_coords:
                print("Nothing to undo.")
            return
        coords, old_vals = action_stack.pop()
        # restore
        rr = [r for r, _ in coords]
        cc = [c for _, c in coords]
        img[rr, cc] = old_vals
        if show_coords:
            print(f"Undid stroke ({len(coords)} px). Undo depth={len(action_stack)}")
        im_artist.set_data(img)
        fig.canvas.draw_idle()

    # ---- Matplotlib setup ----
    fig, ax = plt.subplots()
    ax.set_title("Drag to paint red | Right-click/U=Undo | Q/Enter=Proceed")
    #ax.set_title("Drag to select start of blood decay")
    im_artist = ax.imshow(img,
        interpolation='nearest',  # prevent smoothing
        origin='upper',
        aspect='equal',
    )
    ax.set_axis_off()

    # ---- Event handlers ----
    def on_press(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = int(round(event.ydata))
        c = int(round(event.xdata))
        if not in_bounds(r, c):
            return
        # Left button: begin stroke
        if event.button == 1:
            start_stroke(r, c)
            im_artist.set_data(img)
            fig.canvas.draw_idle()
        # Right button: undo immediately (only if not currently painting)
        elif event.button == 3:
            # If we were somehow mid-stroke and right pressed, end stroke first (unlikely)
            end_stroke()
            undo_last()

    def on_motion(event):
        if not painting:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = int(round(event.ydata))
        c = int(round(event.xdata))
        continue_stroke(r, c)
        im_artist.set_data(img)
        fig.canvas.draw_idle()

    def on_release(event):
        # finish stroke if left button released
        if event.button == 1:
            end_stroke()

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("u", "z", "backspace"):
            # finish any in-progress stroke before undo
            end_stroke()
            undo_last()
        elif key in ("q", "escape", "return") or "enter" in key:
            plt.close(fig)

    # Connect
    cid_press   = fig.canvas.mpl_connect('button_press_event',   on_press)
    cid_motion  = fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    cid_key     = fig.canvas.mpl_connect('key_press_event',      on_key)

    plt.show()  # blocks

    # Clean up
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_key)

    return img, all_painted_points

import cv2
# ---- Example usage ----
if __name__ == "__main__":
    # Simple test image: green background

    def octRecon_angio_Bal_GPU_STFT_window(raw,dispersion,stft_w_g,processParams,kmap):

        try:
            numBands = cp.shape(stft_w_g)[1]
        except:
            numBands = 1
        
        res_fast = processParams.res_fast
        repNum = processParams.repNum
        print("repNum",repNum)
        frame_3D = np.zeros((int(processParams.res_axis/2), res_fast, processParams.res_slow,repNum))
        #frame_3D_out = np.zeros((int(np.shape(raw)[0]), int(processParams.res_axis/2),numBands))
        pdAmt = int((processParams.upSampleFactor-1)*np.shape(raw)[1]/2)   
        #print("pad amount",pdAmt)  
        
        startIndex = range(0,np.shape(raw)[0],res_fast*repNum)
        print("shape of startIndex is",np.shape(startIndex))
        finDepth = int(np.shape(raw)[1]/2)
        count = 0
        for i in range(len(startIndex)):
            
            temp = cp.zeros((res_fast,1024,repNum),dtype = cp.complex128)
            count += 1
            
            if startIndex[i]+res_fast*repNum > np.shape(raw)[0]:
                endIndex = np.shape(raw)[0]
            else:
                endIndex = startIndex[i] + res_fast*repNum
            
            fringe_gpu = cp.asarray(raw[startIndex[i]:endIndex,:])
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

            #for band_id in bands:
                    #print("shape of fringe_lin is", cp.shape(fringe_lin))
                    #print("shape of stft_w_g is:", cp.shape(stft_w_g))
            fringe_lin_win = stft_w_g * fringe_lin
            dispersionPhaseTerm = cp.exp(-1j * dispersion)
            fringe_lin_win = fringe_lin_win * dispersionPhaseTerm
                
            zeroPadAmt = int((processParams.zeroPaddingFactor - 1)*int(cp.shape(fringe_lin_win)[0])/2)
            fringe_lin_win = cp.pad(fringe_lin_win,((0,0),(zeroPadAmt,zeroPadAmt)))

            frame = cp.fft.fft(fringe_lin_win,axis=1)
            frame = frame[:,0:int(cp.shape(frame)[1]/(2*processParams.upSampleFactor))]
            print(cp.shape(frame))
            for j in range(0,repNum):
                temp[:,:,j] = frame[j*int((cp.shape(frame)[0])/repNum):(j+1)*int((cp.shape(frame)[0])/repNum),:]
            temp1 = cp.transpose(temp,(1,0,2))
                #if band_id == 0:
                        #print("shape of fringe_lin_win is",cp.shape(fringe_lin_win))
            temp1 = cp.abs(temp1)
            # for dd in range(1,repNum):
            #     frame1 = temp1[:,:,dd-1]
            #     frame2 = temp1[:,:,dd]
            #     shift = register_images(frame1, frame2, usfac = 1)
            #     temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(shift[1])),0)
            #     temp1[:,:,dd] = np.roll(temp1[:,:,dd],-int(cp.asnumpy(shift[0])),1)
            # fullFrame = cp.squeeze(cp.mean(temp1,axis=2))
            # 1024 x res_fast x res_slow x repNum
            frame_3D[:,:,i,:] = temp1.get()

        #frame_3D_out = frame_3D
        return frame_3D

    raw_fringes = fringes(processParameters.fname,processParameters,processParameters.pixelMap)
            
    class data:
        pass
    wavePath = 'Wavelength Files/wavelength-OH.01.2021.0005-created-28-Jan-2021'
    with open(wavePath, 'rb') as f:
        b = f.read()

    data.wavelength = np.frombuffer(b)
    data.wavelength = data.wavelength*1e-9
    startTime = time.time()
    #pixMap = np.linspace(0,2047,2048)
    if processParameters.balFlag:
        #data.raw = raw_fringes.get_balance_Fringes(interpMat=pixMap)
        data.raw = raw_fringes.get_balance_Fringes()
    else:
        data.raw = raw_fringes.get_unbalance_Fringes()
    endTime = time.time()
    #raw_fringes = None


    linear_wavenumber_matrix = linear_k()

    linear_wavenumber_matrix.calculate_linear_interpolation_matrix(data.wavelength,processParameters)

    class stft:
        pass

    stft.winNum = 19
    stft.resol = 7e-6
    stft.firstWave = 535e-9
    stft.lastWave = 590e-9

    stft.Wavelength_shift = -5e-9

    stft.firstWave = stft.firstWave + stft.Wavelength_shift
    stft.lastWave = stft.lastWave + stft.Wavelength_shift

    stft = calculateWindows(stft,data.wavelength,processParameters,processParameters.pixelMap)

            #plt.plot(stft.stftWin)
            

            # %% Dispersion Compensation
    class dispersion:
        pass
    startTime = time.time()
    if GPU_available:
        stft.stft_w_g = cp.asarray(stft.stftWin)
        linear_wavenumber_matrix.To_GPU()
            #linear_wavenumber_matrix.To_Torch()
    dispersion.c2a, dispersion.c3a = dispersionOptimization_balance(data.raw,data.wavelength,processParameters,processParameters.pixelMap,linear_wavenumber_matrix.kmap,0)
    print(dispersion.c2a)
    print(dispersion.c3a)

    dispersion.dispersion = calcDispersionPhase(linear_wavenumber_matrix.kmap.freq_lin,dispersion.c2a,dispersion.c3a)
    print(1)
    print(1)
    print(1)
    print("idxA")
    print(linear_wavenumber_matrix.kmap.idxA)
    endTime = time.time()
            # %% Basic OCT reconstruction
    startTime = time.time()
    frame_OCTA = None
    octa_lowVal = None
    octa_highVal = None
    if GPU_available and (cp.cuda.runtime.getDeviceProperties(cp.cuda.Device())["totalGlobalMem"] >= 6442450944):
                #Total memory > 6GB
        if processParameters.octaFlag:
            frame_3D_out,frame_OCTA = octangio_Recon_Bal_GPU(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g,processParameters,linear_wavenumber_matrix.kmap)
            
        # for i in range (1,19):
        #     frame_3D_out = octRecon_angio_Bal_GPU_STFT_window(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g[:,i],processParameters,linear_wavenumber_matrix.kmap)
        #     np.save('OCT_stft_window_'+str(i)+'.npy', frame_3D_out)
        else:
            frame_3D_out = octRecon_Bal_GPU(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g[:,0],processParameters,512*4,linear_wavenumber_matrix.kmap)
        cp.get_default_memory_pool().free_all_blocks()
    else:
        frame_3D_out = octRecon_Bal(data.raw,dispersion.dispersion,stft.stftWin[:,0],processParameters,512*2,linear_wavenumber_matrix.kmap)
    endTime = time.time()

    #frame_3D_oct_out = np.reshape(frame_3D_out[:,:,:],(int(processParameters.res_axis/2),cp.shape(stft.stft_w_g)[1],int(processParameters.res_fast),int(processParameters.res_slow)),'A')
    #frame_3D_oct = frame_3D_oct.astype(np.float32)
    #np.save('frame_3D.npy',frame_3D_oct)

    frame_3D_resh = np.reshape(frame_3D_out,(int(processParameters.res_axis/2),int(processParameters.res_fast),int(processParameters.res_slow)),'A')

    if not processParameters.octaFlag:
        print("not OCTA")
        registration.horizontal_flip(frame_3D_resh,processParameters.res_slow)
        frame_3D_resh_GPU = cp.asarray(frame_3D_resh)
        vert_shift = [0]
        backLash = registration.get_backlash_pattern(frame_3D_resh_GPU,int(processParameters.res_slow),vert_shift)
        del frame_3D_resh_GPU
        registration.horizontal_correction(frame_3D_resh,processParameters.res_slow,backLash)
        frame_3D_oct = frame_3D_resh[:,:-np.abs(backLash),:]

    enface = np.squeeze(np.mean(frame_3D_resh,axis=0))
    print("enface shape",np.shape(enface))
    enface = enface/np.mean(enface,axis=0)
    enface = enface/np.max(enface)
    enface = exposure.equalize_adapthist(enface,clip_limit=0.01)
    enface = (enface * 255).astype(np.uint8)
    #enface = cv2.resize(enface,(512,512))
    # print("res slow:",processParameters.res_slow)
    #del frame_3D_out
    #del frame_OCTA
    #del data.raw
    #del raw_fringes
    
    #test_img[..., 1] = 255

    modified, painted_points = interactive_red_painter_ms_paint(enface, brush_radius=1)
    

    for i in range (0,19):
        frame_3D_out = octRecon_angio_Bal_GPU_STFT_window(data.raw,cp.array(dispersion.dispersion),stft.stft_w_g[:,i],processParameters,linear_wavenumber_matrix.kmap)
        np.save('Wills_new_OCT_stft_window_'+str(i)+'.npy', frame_3D_out)
    #painted_points

    aline_stack = np.zeros((1024,processParameters.repNum*len(painted_points),18))
    #print("First 10 points:", painted_points[:10])
    for i in range(0,1):
        v= np.load('Wills_new_OCT_stft_window_'+str(i)+'.npy')
        for j in range(6):
            counter = 0
            for point in painted_points:
                frame_3D_resh = np.reshape(v[:,:,:,j],(int(processParameters.res_axis/2),int(processParameters.res_fast),int(processParameters.res_slow)),'A')
                aline_stack[:,j*counter,i] = frame_3D_resh[:,point[0],point[1]]
                counter += 1

    for i in range(0,1):
        aline_stack[0:,:,i] = aline_reg(cp.asarray(aline_stack[0:,:,i]),cp.asarray(aline_stack[0:,100,i])).get()
    plt.imshow(aline_stack[:,:,0],cmap='gray')
    plt.show()

    sd_aline = np.mean(aline_stack[:,:,0],axis=1)

    from matplotlib.widgets import RectangleSelector
    # Callback function for selection
    def onselect(eclick, erelease):
        x1, x2 = int(eclick.xdata), int(erelease.xdata)
        start, end = sorted([max(0, x1), min(len(sd_aline), x2)])
        selected = sd_aline[start:end]
        print(f"Selected range: [{start}:{end}]")
        print(selected)

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(sd_aline)
    ax.set_title("Drag to select a region")

    # Updated RectangleSelector without 'drawtype'
    selector = RectangleSelector(
        ax, onselect,
        interactive=True,
        useblit=True
    )

    plt.show()
    