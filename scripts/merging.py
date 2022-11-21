import os,sys
import numpy as np 
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
    
import scripts.utils

def resolution_to_radius(resolution_A=None,detectorDistance_mm=None,pixelSize_mm=None,waveLength_A=None):
    radius_reci_invA = 1./waveLength_A
    point_radius_invA = 1./resolution_A
    half_angle_rad = np.arcsin(point_radius_invA/2.0/radius_reci_invA)
    radius_px = np.tan(half_angle_rad*2.0) * detectorDistance_mm / pixelSize_mm
    return radius_px
    
def mapPixel2RealXYZ(size=None, center=None, pixelSize=None, detectorDistance=None,parallax_correction_px=None):
    """
    Input: only 2D size are accepted
    Returns: Real space pixel position (xyz): N*N*3 numpy array
    xyz: unit (meter)
    """
    (nx, ny) = size
    (cx, cy) = center
    if parallax_correction_px is not None:

        xaxis = parallax_correction_px[0]
        yaxis = parallax_correction_px[1]
    else:
        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        xaxis, yaxis = np.meshgrid(x, y, indexing="ij")

    
    zaxis = np.ones((nx,ny))
    xyz = np.stack([xaxis*pixelSize, yaxis*pixelSize, zaxis*detectorDistance], axis=2)
    return xyz


def mapRealXYZ2Reciprocal(xyz=None, waveLength=None):
    """
    The unit of wavelength is A
    xyz: Real space pixel position (xyz): N*N*3 numpy array
    xyz: unit (meter)
    reciprocal: N*N*3 numpy array
    """
    if len(xyz.shape)==2:
        norm = np.sqrt(np.sum(xyz**2, axis=1, keepdims=True))
        ## scaled array doesn't have units
        scaled = xyz/norm
        scaled[:,2] -= 1.0
        reciprocal = scaled / waveLength
        return reciprocal
    elif len(xyz.shape)==3:
        norm = np.sqrt(np.sum(xyz**2, axis=2, keepdims=True))
        ## scaled array doesn't have units
        scaled = xyz/norm
        scaled[:,:,2] -= 1.0
        reciprocal = scaled / waveLength
        return reciprocal
    else:
        return None


def mapReciprocal2Voxel(Amat=None, Bmat=None, returnFormat="HKL", \
                    reciprocal=None, oversample=1, Phi=0., rotAxis="x"):
    """
    Return: voxel (N*N*3 or N*3) 
    voxelSize: 0.015 for 'cartesian' coordinate; 1.0 for "hkl" coordinate
    """
    Phimat = scripts.utils.quaternion2rotation(scripts.utils.phi2quaternion(Phi, rotAxis=rotAxis))
    if returnFormat.lower() == "hkl":
        voxel = reciprocal.dot(np.linalg.inv(Phimat.dot(Amat)).T) * oversample 
    elif returnFormat.lower() == "cartesian":
        voxel = reciprocal.dot(np.linalg.inv(Phimat.dot(Amat)).T).dot(Bmat.T) * oversample 
    return voxel


def mapImage2Voxel(image=None, size=None, Amat=None, Bmat=None, xvector=None, Phi=0., \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                returnFormat=None, oversample=1, rotAxis="x",parallax_correction_px=None):

    """
    # This function combines mapPixel2RealXYZ, mapRealXYZ2Reciprocal and mapReciprocal2Voxel. 
    # Input: real 2D image in N*N
    # Output: voxel in N*N*3 shape
    """
    if image is not None:
        size = image.shape

    if xvector is None:
        xyz = mapPixel2RealXYZ(size=size, center=center, pixelSize=pixelSize, \
                            detectorDistance=detectorDistance,parallax_correction_px=parallax_correction_px)
        reciprocal = mapRealXYZ2Reciprocal(xyz=xyz, waveLength=waveLength)
    else:
        reciprocal = xvector

    voxel = mapReciprocal2Voxel(Amat=Amat, Bmat=Bmat, returnFormat="HKL", rotAxis=rotAxis, \
                            reciprocal=reciprocal, oversample=oversample, Phi=Phi)

    return voxel


@jit
def PeakMask(Amat=None, _image=None, size=None, xvector=None, window=(0, 0.25), \
             hRange=(-1000,1000), kRange=(-1000,1000), lRange=(-1000,1000), \
             waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
             Phi=0., rotAxis="x", parallax_correction_px=None):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    ## peakMask = 1 when pixels are in window
    """
    voxel = mapImage2Voxel(image=_image, size=size, Amat=Amat, xvector=xvector, \
            Phi=Phi, waveLength=waveLength, pixelSize=pixelSize, center=center, rotAxis=rotAxis, \
            detectorDistance=detectorDistance,parallax_correction_px=parallax_correction_px)

    ## For Loop to map one image
    if size is None:
        size = _image.shape

    Npixels = np.prod(size)
    peakMask = np.zeros(Npixels).astype(int)
    voxel =  voxel.reshape((Npixels, 3)) 
    shift = np.abs(np.around(voxel) - voxel)

    for t in range(Npixels):
        
        hshift = shift[t, 0]
        kshift = shift[t, 1]
        lshift = shift[t, 2]

        hh = voxel[t, 0]
        kk = voxel[t, 1]
        ll = voxel[t, 2]

        if (hshift>=window[1]) or (kshift>=window[1]) or (lshift>=window[1]):
            continue
        if (hshift<window[0]) and (kshift<window[0]) and (lshift<window[0]):
            continue
        if hh < hRange[0] or hh >= hRange[1]:
            continue
        if kk < kRange[0] or kk >= kRange[1]:
            continue
        if ll < lRange[0] or ll >= lRange[1]:
            continue
        
        peakMask[t] = 1

    return peakMask.reshape(size)


@jit
def Image2Volume(volume=None, weight=None, Amat=None, Bmat=None, _image=None, _mask=None, \
                keepPeak=False, returnFormat="HKL", xvector=None, \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                Vcenter=60, Vsize=121, oversample=1, Phi=0., rotAxis="x",\
                parallax_correction_px=None,window_keep_hkl=(0.25,1)):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    """

    voxel = mapImage2Voxel(image=_image, size=None, Amat=Amat, Bmat=Bmat, xvector=xvector, \
            Phi=Phi, waveLength=waveLength, pixelSize=pixelSize, center=center, rotAxis=rotAxis, \
            detectorDistance=detectorDistance, returnFormat=returnFormat, oversample=1,\
            parallax_correction_px=parallax_correction_px)

    ## For Loop to map one image
    Npixels = np.prod(_image.shape)
    image = _image.ravel()
    mask  =  _mask.ravel()
    voxel =  voxel.reshape((Npixels, 3)) 
    
    if volume is None:
        volume = np.zeros((Vsize, Vsize, Vsize))
        weight = np.zeros((Vsize, Vsize, Vsize))
        
    for t in range(Npixels):

        if mask[t] == 0:
            continue
        
        hkl = voxel[t]
        
        real_h = hkl[0] 
        real_k = hkl[1] 
        real_l = hkl[2] 
        
        int_h = int(round(real_h)) 
        int_k = int(round(real_k)) 
        int_l = int(round(real_l)) 
        
        hshift = abs( real_h - int_h )
        kshift = abs( real_k - int_k )
        lshift = abs( real_l - int_l )

        if (hshift>=window_keep_hkl[1]) or (kshift>=window_keep_hkl[1]) or (lshift>=window_keep_hkl[1]):
            continue
        if (hshift<window_keep_hkl[0]) and (kshift<window_keep_hkl[0]) and (lshift<window_keep_hkl[0]):
            continue
            
        save_h = int(round(real_h*oversample + Vcenter) ) 
        save_k = int(round(real_k*oversample + Vcenter) ) 
        save_l = int(round(real_l*oversample + Vcenter) ) 
        
        if (save_h<0) or save_h>(Vsize-1) or (save_k<0) or save_k>(Vsize-1) or (save_l<0) or save_l>(Vsize-1):
            continue
        
        weight[save_h,save_k,save_l] += 1
        volume[save_h,save_k,save_l] += image[t]

    return volume, weight


def mapImage2Resolution(image=None, size=None, waveLength=None, detectorDistance=None, detectorCenter=None, pixelSize=None, format="res"):
    if image is not None:
        size = image.shape

    xyz = mapPixel2RealXYZ(size=size, center=detectorCenter, pixelSize=pixelSize, detectorDistance=detectorDistance)
    rep = mapRealXYZ2Reciprocal(xyz=xyz, waveLength=waveLength)
    repNorm = np.sqrt(np.sum(rep**2, axis=2))

    if format == "res":
        res = np.zeros(size)
        index = np.where(repNorm>0)
        res[index] = 1./repNorm[index]
        res[repNorm==0] = np.amax(res)
        return res
    else:
        return repNorm
    
def radius_px_to_resolution_A(radius_px=None,pixel_size_mm=None,detector_distance_mm=None,wavelength_A=None):
    # radius_px is array
    index = np.where(radius_px==0)
    radius_px[index] = 1.0e-10
    tan_theta = radius_px * pixel_size_mm * 1.0 / detector_distance_mm
    half_angle_rad = np.arctan(tan_theta)/2.0
    reciprocal_invA = 1.0 / wavelength_A * np.sin(half_angle_rad) * 2.
    return 1./reciprocal_invA