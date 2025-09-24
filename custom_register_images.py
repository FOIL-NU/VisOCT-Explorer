
from cupy.fft import fftn, ifftn, fftfreq
import numpy as np
import cupy as cp
from cupy.fft import ifftshift,fftfreq
from cupy import pi,newaxis,floor,dot

__all__ = ['register_images']


def register_images(im1, im2, usfac=1, return_registered=False,
                    return_error=False, zeromean=True, DEBUG=False, maxoff=None,
                    nthreads=1, use_numpy_fft=False):
    """
    Sub-pixel image registration (see dftregistration for lots of details)
    Parameters
    ----------
    im1 : np.ndarray
    im2 : np.ndarray
        The images to register.
    usfac : int
        upsampling factor; governs accuracy of fit (1/usfac is best accuracy)
    return_registered : bool
        Return the registered image as the last parameter
    return_error : bool
        Does nothing at the moment, but in principle should return the "fit
        error" (it does nothing because I don't know how to compute the "fit
        error")
    zeromean : bool
        Subtract the mean from the images before cross-correlating?  If no, you
        may get a 0,0 offset because the DC levels are strongly correlated.
    maxoff : int
        Maximum allowed offset to measure (setting this helps avoid spurious
        peaks)
    DEBUG : bool
        Test code used during development.  Should DEFINITELY be removed.
    Returns
    -------
    dx,dy : float,float
        REVERSE of dftregistration order (also, signs flipped) for consistency
        with other routines.
        Measures the amount im2 is offset from im1 (i.e., shift im2 by these #'s
        to match im1)
    """
    if not im1.shape == im2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        im1 = im1 - (im1[im1==im1].mean())
        im2 = im2 - (im2[im2==im2].mean())

    if cp.any(cp.isnan(im1)):
        im1 = im1.copy()
        im1[im1!=im1] = 0
    if cp.any(cp.isnan(im2)):
        im2 = im2.copy()
        im2[im2!=im2] = 0

    #fft2,ifft2 = fftn,ifftn

    im1fft = fftn(im1)
    im2fft = fftn(im2)

    output = dftregistration(im1fft,im2fft,usfac=usfac,
                             return_registered=return_registered,
                             return_error=return_error, zeromean=zeromean,
                             DEBUG=DEBUG, maxoff=maxoff)

    output = [-output[1], -output[0], ] + [o for o in output[2:]]

    if return_registered:
        output[-1] = cp.abs(cp.fft.ifftshift(ifftn(output[-1])))
    #del im1,im2
    return output


def dftregistration(buf1ft, buf2ft, usfac=1, return_registered=False,
                    return_error=False, zeromean=True, DEBUG=False, maxoff=None,
                    nthreads=1, use_numpy_fft=False):
    """
    translated from matlab:
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html
    Efficient subpixel image registration by crosscorrelation. This code
    gives the same precision as the FFT upsampled cross correlation in a
    small fraction of the computation time and with reduced memory
    requirements. It obtains an initial estimate of the crosscorrelation peak
    by an FFT and then refines the shift estimation by upsampling the DFT
    only in a small neighborhood of that estimate by means of a
    matrix-multiply DFT. With this procedure all the image points are used to
    compute the upsampled crosscorrelation.
    Manuel Guizar - Dec 13, 2007
    Portions of this code were taken from code written by Ann M. Kowalczyk
    and James R. Fienup.
    J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
    object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
    (1990).
    Citation for this algorithm:
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    "Efficient subpixel image registration algorithms," Opt. Lett. 33,
    156-158 (2008).
    Inputs
    buf1ft    Fourier transform of reference image,
           DC in (1,1)   [DO NOT FFTSHIFT]
    buf2ft    Fourier transform of image to register,
           DC in (1,1) [DO NOT FFTSHIFT]
    usfac     Upsampling factor (integer). Images will be registered to
           within 1/usfac of a pixel. For example usfac = 20 means the
           images will be registered within 1/20 of a pixel. (default = 1)
    Outputs
    output =  [error,diffphase,net_row_shift,net_col_shift]
    error     Translation invariant normalized RMS error between f and g
    diffphase     Global phase difference between the two images (should be
               zero if images are non-negative).
    net_row_shift net_col_shift   Pixel shifts between images
    Greg      (Optional) Fourier transform of registered version of buf2ft,
           the global phase difference is compensated for.
    """

    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from cupy import conj,abs,arctan2,sqrt,real,imag,shape,zeros,trunc,ceil,floor,fix,sum
    from cupy.fft import fftshift,ifftshift

    # Compute error for no pixel shift
    if usfac == 0:
        raise ValueError("Upsample Factor must be >= 1")
        CCmax = sum(sum(buf1ft * conj(buf2ft)));
        rfzero = sum(abs(buf1ft)**2);
        rgzero = sum(abs(buf2ft)**2);
        error = 1.0 - CCmax * conj(CCmax)/(rgzero*rfzero);
        error = sqrt(abs(error));
        diffphase=arctan2(imag(CCmax),real(CCmax));
        output=[error,diffphase];

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    # peak
    elif usfac == 1:
        [m,n]=shape(buf1ft)
        CC = ifftn(buf1ft * conj(buf2ft))
        if maxoff is None:
            rloc,cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]
        else:
            # set the interior of the shifted array to zero
            # (i.e., ignore it)
            CC[maxoff:-maxoff,:] = 0
            CC[:,maxoff:-maxoff] = 0
            rloc,cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]
        rfzero = sum(abs(buf1ft)**2)/(m*n)
        rgzero = sum(abs(buf2ft)**2)/(m*n)
        error = 1.0 - CCmax * conj(CCmax)/(rgzero*rfzero)
        error = sqrt(abs(error))
        diffphase=arctan2(imag(CCmax),real(CCmax))
        md2 = fix(m/2)
        nd2 = fix(n/2)
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc
        #output=[error,diffphase,row_shift,col_shift];
        output=[row_shift,col_shift]

    # Partial-pixel shift
    else:

        if DEBUG: import pylab
        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        [m,n]=shape(buf1ft)
        mlarge=m*2
        nlarge=n*2
        CClarge=zeros([mlarge,nlarge], dtype='complex')
        #CClarge[m-fix(m/2):m+fix((m-1)/2)+1,n-fix(n/2):n+fix((n-1)/2)+1] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        CClarge[int(m-np.fix(m/2)):int(m+np.fix((m-1)/2)+1),int(n-np.fix(n/2)):int(n+np.fix((n-1)/2)+1)] = fftshift(buf1ft) * conj(fftshift(buf2ft))
        # note that matlab uses fix which is trunc... ?

        # Compute crosscorrelation and locate the peak
        CC = ifftn(ifftshift(CClarge)); # Calculate cross-correlation
        if maxoff is None:
            rloc,cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]
        else:
            # set the interior of the shifted array to zero
            # (i.e., ignore it)
            CC[maxoff:-maxoff,:] = 0
            CC[:,maxoff:-maxoff] = 0
            rloc,cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]

        if DEBUG:
            pylab.figure(1)
            pylab.clf()
            pylab.subplot(131)
            pylab.imshow(real(CC)); pylab.title("Cross-Correlation (upsampled 2x)")
            pylab.subplot(132)
            ups = dftups((buf1ft) * conj((buf2ft)),mlarge,nlarge,2,0,0); pylab.title("dftups upsampled 2x")
            pylab.imshow(real(((ups))))
            pylab.subplot(133)
            pylab.imshow(real(CC)/real(ups)); pylab.title("Ratio upsampled/dftupsampled")
            print("Upsample by 2 peak: ",rloc,cloc," using dft version: ",cp.unravel_index(abs(ups).argmax(), ups.shape))
            #print np.unravel_index(ups.argmax(),ups.shape)

        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak
        [m,n] = shape(CC); md2 = trunc(m/2); nd2 = trunc(n/2)
        if rloc > md2 :
            row_shift2 = rloc - m
        else:
            row_shift2 = rloc
        if cloc > nd2:
            col_shift2 = cloc - n
        else:
            col_shift2 = cloc
        row_shift2=row_shift2/2.
        col_shift2=col_shift2/2.
        

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            #%% DFT computation %%%
            # Initial shift estimate in upsampled grid
            zoom_factor=1.5
            
            row_shift0 = cp.round(row_shift2*usfac)/usfac
            col_shift0 = cp.round(col_shift2*usfac)/usfac
            dftshift = trunc(ceil(usfac*zoom_factor)/2); #% Center of output array at dftshift+1
            
            # Matrix multiply DFT around the current shift estimate
            roff = dftshift-row_shift0*usfac
            coff = dftshift-col_shift0*usfac
            inp = buf2ft * conj(buf1ft)

            #upsampled = dftups(inp,ceil(usfac*zoom_factor),ceil(usfac*zoom_factor),usfac,roff,coff)

            nr,nc=cp.shape(inp)
            nor = ceil(usfac*zoom_factor)
            noc = ceil(usfac*zoom_factor)
            # Compute kernels and obtain DFT by matrix products
            term1c = ( ifftshift(cp.arange(nc,dtype='float') - floor(nc/2)).T[:,newaxis] )/nc # fftfreq
            term2c = (( cp.arange(noc,dtype='float') - coff  )/usfac)[newaxis,:]              # output points
            kernc=cp.exp((-1j*2*pi)*term1c*term2c)

            term1r = ( cp.arange(nor,dtype='float').T - roff )[:,newaxis]                # output points
            term2r = ( ifftshift(cp.arange(nr,dtype='float')) - floor(nr/2) )[newaxis,:] # fftfreq
            kernr=cp.exp((-1j*2*pi/(nr*usfac))*term1r*term2r)
            upsampled=cp.dot(cp.dot(kernr,inp),kernc)

            #CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
            #    dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac^2);
            CC = conj(upsampled)/(md2*nd2*usfac**2)
            if DEBUG:
                pylab.figure(2)
                pylab.clf()
                pylab.subplot(221)
                pylab.imshow(abs(upsampled)); pylab.title('upsampled')
                pylab.subplot(222)
                pylab.imshow(abs(CC)); pylab.title('CC upsampled')
                pylab.subplot(223); pylab.imshow(cp.abs(cp.fft.fftshift(cp.fft.ifft2(buf2ft * conj(buf1ft))))); pylab.title('xc')
                yy,xx = np.indices([m*usfac,n*usfac],dtype='float')
                pylab.contour(yy/usfac/2.-0.5+1,xx/usfac/2.-0.5-1, cp.abs(dftups((buf2ft*conj(buf1ft)),m*usfac,n*usfac,usfac)))
                pylab.subplot(224); pylab.imshow(cp.abs(dftups((buf2ft*conj(buf1ft)),ceil(usfac*zoom_factor),ceil(usfac*zoom_factor),usfac))); pylab.title('unshifted ups')
            # Locate maximum and map back to original pixel grid
            rloc,cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)
            rloc0,cloc0 = cp.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax = CC[rloc,cloc]
            #[max1,loc1] = CC.max(axis=0), CC.argmax(axis=0)
            #[max2,loc2] = max1.max(),max1.argmax()
            #rloc=loc1[loc2];
            #cloc=loc2;
            #CCmax = CC[rloc,cloc];
            rg00 = dftups(buf1ft * conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2)
            rf00 = dftups(buf2ft * conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2)
            #if DEBUG: print rloc,row_shift,cloc,col_shift,dftshift
            rloc = rloc - dftshift #+ 1 # +1 # questionable/failed hack + 1;
            cloc = cloc - dftshift #+ 1 # -1 # questionable/failed hack - 1;
            #if DEBUG: print rloc,row_shift,cloc,col_shift,dftshift
            row_shift = row_shift0 + rloc/usfac
            col_shift = col_shift0 + cloc/usfac
            del upsampled
            #if DEBUG: print rloc/usfac,row_shift,cloc/usfac,col_shift
            if DEBUG: print("Off by: ",(0.25 - float(rloc)/usfac)*usfac , (-0.25 - float(cloc)/usfac)*usfac )
            if DEBUG: print("correction was: ",rloc/usfac, cloc/usfac)
            if DEBUG: print("Coordinate went from",row_shift2,col_shift2,"to",row_shift0,col_shift0,"to", row_shift, col_shift)
            if DEBUG: print("dftsh - usfac:", dftshift-usfac)
            if DEBUG: print( rloc,cloc,row_shift,col_shift,CCmax,dftshift,rloc0,cloc0)

        # If upsampling = 2, no additional pixel shift refinement
        else:
            rg00 = sum(sum( buf1ft * conj(buf1ft) ))/m/n
            rf00 = sum(sum( buf2ft * conj(buf2ft) ))/m/n
            row_shift = row_shift2
            col_shift = col_shift2
        error = 1.0 - CCmax * conj(CCmax)/(rg00*rf00)
        error = sqrt(abs(error))
        diffphase=arctan2(imag(CCmax),real(CCmax))
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0
        if nd2 == 1:
            col_shift = 0
        #output=[error,diffphase,row_shift,col_shift];
        output=[row_shift,col_shift]      #change back to row_shift and col_shift


    if return_error:
        # simple estimate of the precision of the fft approach
        output += [1./usfac,1./usfac]

    # Compute registered version of buf2ft
    if (return_registered):
        if (usfac > 0):
            nr,nc=shape(buf2ft)
            Nr = cp.fft.ifftshift(cp.linspace(-np.fix(nr/2),cp.ceil(nr/2)-1,nr))
            Nc = cp.fft.ifftshift(cp.linspace(-np.fix(nc/2),cp.ceil(nc/2)-1,nc))
            [Nc,Nr] = cp.meshgrid(Nc,Nr)
            Greg = buf2ft * cp.exp(1j*2*cp.pi*(-row_shift*Nr/nr-col_shift*Nc/nc))
            Greg = Greg*cp.exp(1j*diffphase)
        elif (usfac == 0):
            Greg = buf2ft*cp.exp(1j*diffphase)
        output.append(Greg)



    return output

def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
    """
    Translated from matlab:
     * `Original Source <http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html>`_
     * Manuel Guizar - Dec 13, 2007
     * Modified from dftus, by J.R. Fienup 7/31/06
    Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    a small region.
    This code is intended to provide the same result as if the following
    operations were performed:
      * Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      * Take the FFT of the larger array
      * Extract an [nor, noc] region of the result. Starting with the
        [roff+1 coff+1] element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
    Parameters
    ----------
    usfac : int
        Upsampling factor (default usfac = 1)
    nor,noc : int,int
        Number of pixels in the output upsampled DFT, in units of upsampled
        pixels (default = size(in))
    roff, coff : int, int
        Row and column offsets, allow to shift the output array to a region of
        interest on the DFT (default = 0)
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab

    nr,nc=cp.shape(inp)
    # Set defaults
    if noc is None: noc=nc
    if nor is None: nor=nr
    # Compute kernels and obtain DFT by matrix products
    term1c = ( ifftshift(cp.arange(nc,dtype='float') - floor(nc/2)).T[:,newaxis] )/nc # fftfreq
    term2c = (( cp.arange(noc,dtype='float') - coff  )/usfac)[newaxis,:]              # output points
    kernc=cp.exp((-1j*2*pi)*term1c*term2c)

    term1r = ( cp.arange(nor,dtype='float').T - roff )[:,newaxis]                # output points
    term2r = ( ifftshift(cp.arange(nr,dtype='float')) - floor(nr/2) )[newaxis,:] # fftfreq
    kernr=cp.exp((-1j*2*pi/(nr*usfac))*term1r*term2r)
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=cp.asarray(dot(dot(kernr,inp),kernc))
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    del term1c
    del term2c
    del kernc
    del term1r
    del term2r
    del kernr

    return out