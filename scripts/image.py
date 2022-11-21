import numpy as np
from numba import jit
from scipy.ndimage.filters import median_filter

class MaskTools:
    def circleMask(self, size, rmin=None, rmax=None, center=None):
        """
        # keep pixels within [rmin,rmax]
        """
        (nx, ny) = size
        if center is None: 
            cx=(nx-1.)/2.
            cy=(ny-1.)/2.
        else:
            (cx,cy) = center
        if rmin is None:
            rmin = -1
        if rmax is None:
            rmax = max(size)*2.

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        [xaxis, yaxis] = np.meshgrid(x,y, indexing="ij")
        r = np.sqrt(xaxis**2+yaxis**2)

        mask = np.zeros(size).astype(int)
        index = np.where((r <= rmax) & (r >= rmin))
        mask[index] = 1
        return mask

    def valueLimitMask(self, image, vmin=None, vmax=None):
        """
        keep pixels value within [vmin,vmax]
        """ 
        mask = np.zeros(image.shape).astype(int)
        (_vmin, _vmax) = (np.amin(image), np.amax(image))
        if vmin is None:
            vmin = _vmin - 1
        if vmax is None:
            vmax = _vmax + 1
        index = np.where((image>=vmin) & (image<=vmax))
        mask[index] = 1
        return mask

    def expandMask(self, mask, expandSize=(1,1), expandValue=0):
        """
        expand mask=expandValue larger
        """
        (nx,ny) = mask.shape
        newMask = mask.copy()
        index = np.where(mask==expandValue)
        for i in range(-expandSize[0], expandSize[0]+1):
            for j in range(-expandSize[1], expandSize[1]+1):
                newMask[((index[0]+i)%nx, (index[1]+j)%ny)] = expandValue
        return newMask


class ScaleTools:
    def solid_angle_scaler(self, size=None, parallax_corrected_position=None, detectorDistance=None, pixelSize=None, center=None):
        """
        Params: detectorDistance, pixelSize must have the same unit
        Returns: scaleMask -> image *= scaleMask (multiply_scaler)
        Note: min(scaleMask) = 1
        """
        (nx, ny) = size
        (cx, cy) = center
        if parallax_corrected_position:
            xaxis, yaxis = parallax_corrected_position
        else:
            x = np.arange(nx) - cx
            y = np.arange(ny) - cy
            xaxis, yaxis = np.meshgrid(x, y, indexing="ij") 
        
#         print xaxis[100,100], yaxis[100,100]
        zaxis = np.ones((nx,ny))*detectorDistance/pixelSize
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        solidAngle = zaxis * 1.0 / norm**3
        solidAngle /= np.amax(solidAngle)
        scaleMask = 1./solidAngle
        return scaleMask

    def polarization_scaler(self, size=None, parallax_corrected_position=None, polarization=-1, detectorDistance=None, pixelSize=None, center=None):
        """
        p =1 means y polarization
        p=-1 means x polarization
        # Default is p=-1 (x polarization)
        # Note: scaleMask -> min=1
        """
        (nx, ny) = size
        (cx, cy) = center
        if parallax_corrected_position:
            xaxis, yaxis = parallax_corrected_position
        else:
            x = np.arange(nx) - cx
            y = np.arange(ny) - cy
            xaxis, yaxis = np.meshgrid(x, y, indexing="ij") 
            
        zaxis = np.ones((nx,ny))*detectorDistance/pixelSize
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
#         print xaxis[100,100], yaxis[100,100]
        if polarization is not None:
            detectScale = (2.*zaxis**2 + (1+polarization)*xaxis**2 + (1-polarization)*yaxis**2 )/(2.*norm**2)
        else: 
            detectScale = np.ones(size)

        detectScale /= np.amax(detectScale)
        scaleMask = 1. / detectScale
        return scaleMask

    def detector_absorption_scaler(self,size,center_px=None,detectorDistance_mm=None,parallax_corrected_position_px=None,pixelSize_um=None,\
                                   thickness_um=None,absorption_coefficient_invum=None,wavelength_A=None):
        # image = image * scaler
        (nx, ny) = size
        if parallax_corrected_position_px is None:
            x = np.arange(nx) - center_px[0]
            y = np.arange(ny) - center_px[1]
            xaxis, yaxis = np.meshgrid(x, y, indexing="ij")
        else:
            xaxis, yaxis = parallax_corrected_position_px
        zaxis = np.ones((nx,ny))*detectorDistance_mm*1000./pixelSize_um
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
#         print xaxis[100,100], yaxis[100,100]
        cos_angle = zaxis / norm
        absorbed_rate = 1. - np.exp( - thickness_um * absorption_coefficient_invum / cos_angle)
        absorbed_rate /= np.amax(absorbed_rate)
        return 1./absorbed_rate
    
    def parallax_correction(self,size,detectorDistance_mm=None,pixelSize_um=None,center_px=None,thickness_um=None,wavelength_A=None,absorption_coefficient_invum=None):
        ## corr_x_px: corrected pixel position in x axis 
        ## corr_y_px: corrected pixel position in y axis 
        (nx, ny) = size
        x = np.arange(nx) - center_px[0]
        y = np.arange(ny) - center_px[1]
        xaxis_px, yaxis_px = np.meshgrid(x, y, indexing="ij") 
        vector = np.zeros((nx,ny,3))
        vector[:,:,0] = xaxis_px
        vector[:,:,1] = yaxis_px
        vector[:,:,2] = np.ones((nx,ny))*detectorDistance_mm*1000./pixelSize_um
#         print vector.shape
        vector /= np.linalg.norm(vector,axis=2,keepdims=2)
        
        atten_um = 1./absorption_coefficient_invum - (thickness_um*1.0/vector[:,:,2] + 1.0/absorption_coefficient_invum)*np.exp(-1.0*absorption_coefficient_invum*thickness_um/vector[:,:,2])
        corr_x_px = xaxis_px - atten_um * vector[:,:,0] / pixelSize_um
        corr_y_px = yaxis_px - atten_um * vector[:,:,1] / pixelSize_um
        return corr_x_px, corr_y_px

    

import scipy.ndimage
from scipy.ndimage import median_filter

class FilterTools:
    def median_filter(self, image=None, mask=None, window=(11,11)):
        ## return data has non zero value even for mask=0
#         print window
        nx,ny = image.shape 
        if mask is not None:
            image_bak = image.copy()
            max_ads_image = int(np.amax(np.abs(image)))
            dark_value = max_ads_image + 10
            chessboard = (np.arange(nx)[:, None] + np.arange(ny)) % 2 == 0
            chessboard = (chessboard.astype(int)*2-1)*dark_value
            index = np.where(mask==0)
            image_bak[index] = chessboard[index]
            image_bak = median_filter(image_bak,window,mode='mirror')
            index = np.where(np.abs(image_bak)>max_ads_image)
            mask[index] = 0
            image_bak[index] = 0
            return image_bak
        else:
            return median_filter(image,window,mode='mirror') 

    def mean_filter(self, image=None,mask=None,window=(11,11)): 
        kernel = np.ones(window)
        if mask is not None:
            sum_image = scipy.ndimage.convolve(image * mask, kernel, mode="mirror")
            sum_mask = scipy.ndimage.convolve(mask, kernel, mode="mirror")
            index = np.where(sum_mask>0)
            sum_image[index] /= 1.0 * sum_mask[index]
            return sum_image * mask 
        else:
            sum_image = scipy.ndimage.convolve(image, kernel, mode="mirror")
            return sum_image * 1.0 / np.prod(window)

    def std_filter(self,image=None,mask=None,window=(11,11)):
        # sigma = sqrt( E(x^2) - E(x)^2 ) 
        kernel = np.ones(window)
        if mask is not None:
            sum_image  = scipy.ndimage.convolve(image * mask,    kernel, mode="mirror")
            sum_square = scipy.ndimage.convolve(image**2 * mask, kernel, mode="mirror") 
            sum_mask   = scipy.ndimage.convolve(mask,            kernel, mode="mirror")
            index = np.where(sum_mask>0)
            sum_image[index] /= 1.0 * sum_mask[index]
            sum_square[index] /= 1.0 * sum_mask[index]
            return np.sqrt( sum_square*mask - (sum_image*mask)**2)
        else:
            sum_image  = scipy.ndimage.convolve(image ,    kernel, mode="mirror")
            sum_square = scipy.ndimage.convolve(image**2,  kernel, mode="mirror")
            return np.sqrt( sum_square*1./np.prod(window) - (sum_image*1./np.prod(window))**2 )


Masktbk = MaskTools()
Filtertbx = FilterTools()
def removeExtremes(_image=None, algorithm=2, _mask=None, _sigma=15, _vmin=None, _vmax=None, _window=(11,11)):
    """
    1. Throw away {+,-}sigma*std from (image - median)
    2. Throw away {+,-}sigma*std again from (image - median)
    """
    if algorithm == 1:
        if _mask is None: 
            mask=np.ones(_image.shape).astype(int)
        else: 
            mask = _mask.astype(int)
        
        image = _image * mask
        mt = MaskTools()
        ft = FilterTools()

        ## remove value >vmax or <vmin
        mask  *= mt.valueLimitMask(image, vmin=_vmin, vmax=_vmax)
        image *= mask
        
        ## remove values {+,-}sigma*std
        median = ft.median_filter(image=image, mask=mask, window=_window)
        submedian = image - median
        # tmp* submedian *= mask
        
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>ave+std*_sigma) | (submedian<ave-std*_sigma))
        image[index] = 0
        mask[index] = 0
        submedian[index] = 0
        
        ## remove values {+,-}sigma*std
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>ave+std*_sigma) | (submedian<ave-std*_sigma))
        image[index] = 0
        mask[index] = 0
        
        return image, mask
    elif algorithm == 2:
        image=_image.copy()
        mask=_mask.copy()
        sigma=_sigma
        vmin=_vmin
        vmax=_vmax
        window=_window
#         print window
        if mask is not None:
            imask = mask.copy() * Masktbk.valueLimitMask(image, vmin=vmin, vmax=vmax)
        else:
            imask = np.ones(image.shape).astype(int) * Masktbk.valueLimitMask(image, vmin=vmin, vmax=vmax)
            
        idata = image * imask
        mean_idata = Filtertbx.mean_filter(image=idata,mask=imask,window=window)
        std_idata  = Filtertbx.std_filter(image=idata,mask=imask,window=window)
        imask *= (idata >= (mean_idata - sigma*std_idata))
        imask *= (idata <= (mean_idata + sigma*std_idata))
        return idata * imask, imask
    else:
        return None,None

@jit
def angularDistri(arr, Arange=None, num=30, rmax=None, rmin=None, center=(None,None)):
    """
    # num denotes how many times you want to divide the full angle (360 degree)
    # This function is slow because it applies multiple for loops
    """
    assert len(arr.shape)==2
    (nx, ny) = arr.shape
    cx = center[0];
    cy = center[1];
    if cx is None: cx = (nx-1.)/2.
    if cy is None: cy = (ny-1.)/2.

    xaxis = np.arange(nx)-cx + 1.0e-5; 
    yaxis = np.arange(ny)-cy + 1.0e-5; 
    [x,y] = np.meshgrid(xaxis, yaxis, indexing='ij')
    r = np.sqrt(x**2+y**2)
    sinTheta = y/r;
    cosTheta = x/r; 
    angle = np.arccos(cosTheta);
    index = np.where(sinTheta<0);
    angle[index] = 2*np.pi - angle[index]
    if rmin is not None:
        index = np.where(r<rmin);
        angle[index] = -1
    if rmax is not None:
        index = np.where(r>rmax);
        angle[index] = -1
    if Arange is not None:
        index = np.where((angle>Arange[0]*np.pi/180.)*(angle<Arange[1]*np.pi/180.)==True);
        subData = arr[index].copy()
        aveIntens = np.mean(subData)
        aveAngle = (Arange[0]+Arange[1])/2.        
        return [aveAngle, aveIntens];

    rad = np.linspace(0, 2*np.pi, num+1);
    aveIntens = np.zeros(num)
    aveAngle = np.zeros(num)
    for i in range(num):
        index = np.where((angle>rad[i])*(angle<rad[i+1])==True);
        subData = arr[index].copy()
        aveIntens[i] = np.mean(subData)
        aveAngle[i] = (rad[i]+rad[i+1])/2.
    return [aveAngle, aveIntens]

@jit
def radialProfile(image, mask, center=None, vmin=None, vmax=None, rmin=None, rmax=None, stepSize=None, sampling=None, window=3):
    """
    # mask = 0 will be ignored
    # pixel value beyond (vmin, vmax) will be ignored
    # radius beyong (rmin, rmax) will be ignored
    # stepSize=1 is normally set
    # sampling is the number of radius points to collect
    # if stepSize is set by user, then sampling will be ignored
    # returns: aveRadius, aveIntens, sumCount
    # checked +1,
    """
    (nx, ny) = image.shape
    if center is None: 
        cx = (nx-1.)/2.
        cy = (ny-1.)/2.
    else:
        cx = center[0]
        cy = center[1]

    x = np.arange(nx)-cx 
    y = np.arange(ny)-cy 
    xaxis,yaxis = np.meshgrid(x, y, indexing='ij')
    radius = np.sqrt(xaxis**2+yaxis**2)

    if rmin is None: 
        rmin = np.amin(radius)
    if rmax is None:
        rmax = np.amax(radius)
    if stepSize is not None:
        aveRadius = np.arange(rmin, rmax+stepSize/2., stepSize)
    elif sampling is not None:
        aveRadius = np.linspace(rmin, rmax, sampling)
        stepSize = (rmax - rmin)/(sampling - 1.)
    else:
        aveRadius = np.arange( int(round(rmin)), int(round(rmax))+1 )
        stepSize = 1.
        
    # print "stepSize = ",np.around(stepSize,2)
    # print "rmin/rmax = ", np.around(aveRadius[0]),np.around(aveRadius[-1])

    notation = mask.copy()
    notation[radius < rmin] = 0
    notation[radius >= rmax] = 0 
    if vmax is not None:
        notation[image >= vmax] = 0   
    if vmin is not None:
        notation[image < vmin] = 0
    
    radius = np.around(radius / stepSize).astype(int)
    startR = int(np.around(rmin / stepSize))
    sumIntens = np.zeros(len(aveRadius))
    sumCount  = np.zeros(len(aveRadius))
    aveIntens = np.zeros(len(aveRadius))

    hwindow = int((window-1)/2.)
    min_radius_has_value = np.amin(radius[notation==1])
    max_radius_has_value = np.amax(radius[notation==1])
    
    for idx in range(nx):
        for jdx in range(ny):
            r = radius[idx, jdx]
            if notation[idx, jdx] == 0:
                continue
            #sumIntens[r-startR] += image[idx,jdx] * notation[idx,jdx]
            #sumCount[r-startR] += notation[idx,jdx]
            
            for h in range(-hwindow, hwindow+1):
                if r - startR + h <= len(sumIntens)-1 and r - startR + h >= 0:
                    sumIntens[r-startR+h] += image[idx,jdx] * notation[idx,jdx]
                    sumCount[r-startR+h] += notation[idx,jdx]     
            
    index = np.where(sumCount > 10)
    aveIntens[index] = sumIntens[index] * 1.0 / sumCount[index]
    index = np.where(aveRadius<min_radius_has_value)
    aveIntens[index] = 0
    index = np.where(aveRadius>max_radius_has_value)
    aveIntens[index] = 0
    return aveRadius, aveIntens, sumCount