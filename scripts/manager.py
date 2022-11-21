import copy
import os,sys
import numpy as np
from numba import jit
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.image
import scripts.merging
import scripts.fsystem
import scripts.datafile

MaskTools   = scripts.image.MaskTools()
ScaleTools  = scripts.image.ScaleTools()
FilterTools = scripts.image.FilterTools()
from scipy.ndimage.filters import median_filter


## data structure
class DataStruct(object):
    def __init__(self):
        self.image = None
        self.mask = 1
        self.peakMask = None
        self.exposure_time_s = None # second
        self.phi_deg = None          # deg
        self.wavelength_A = None   # A
        self.pixel_size_mm = None    # mm 
        self.pixel_size_um = None    # um 
        self.polarization_fr = -1   
        self.detector_distance_mm = None  # mm 
        self.detector_center_px = None    # px
        self.lattice_constant_A_deg = None   # A, deg
        self.thickness_um = None      # um
        self.thickness_mm = None      # mm
        self.absorption_coefficient_invum = None  # um^-1
        self.absorption_coefficient_invmm = None  # um^-1
        self.rotAxis = "x"
        self.Amat_invA = None              # A^-1
        self.Bmat_invA = None              # A^-1


## Information of a single diffraction pattern
class ImageAgent(DataStruct):
    def __init__(self, filename=None):
        DataStruct.__init__(self)
        if filename is not None:
            self.loadImage(filename)
        if self.image is not None:
            if isinstance(self.mask,(int,float)):
                self.mask = np.ones(self.image.shape).astype(int)
                
        
    def todict(self):
        return self.__dict__
    
    def fromdict(self, entry):
        self.__dict__.update(entry)
    
    def fromobject(self, classobject):
        self = copy.deepcopy(classobject)
        
    def readfile(self, filename=None, fileType=None):
        ## return data in psvm format 
        if filename is None:
            return {}
        psvm = scripts.datafile.reader(filename, fileType=fileType)
        if psvm is None:
            return {}
        return psvm
    
    def loadImage(self, filename, fileType=None):
        psvm = self.readfile(filename, fileType=fileType)
        self.fromdict(psvm)
        return True
    
    def apply_detector_mask(self,radius_rmin_px=40,radius_rmax_px=None,value_vmin=0,value_vmax=10000,**kwargs):
#         print radius_rmin_px,radius_rmax_px,value_vmin,value_vmax
        if hasattr(self,"detector_mask_file"):
            self.mask *= np.load(self.detector_mask_file) 
        self.mask *= MaskTools.valueLimitMask(self.image, vmin=value_vmin, vmax=value_vmax)
        self.expandMask(expand_mask_px=1)
        self.mask *= MaskTools.circleMask(self.image.shape, \
                                          rmin=radius_rmin_px, \
                                          rmax=radius_rmax_px, \
                                          center=self.detector_center_px)
        
        self.image *= self.mask 
        
    def remove_bad_pixels(self,algorithm=2,sigma=5,window_size_px=11,**kwargs):
#         print algorithm,sigma,window_size_px
        newImage, newMask = scripts.image.removeExtremes(_image=self.image,_mask=self.mask, \
                                                        algorithm=algorithm,_sigma=sigma, \
                                                        _window=(window_size_px,window_size_px))
        
        self.image = newImage.copy()
        self.mask *= newMask
        self.image *= self.mask 
        
    def subtract_background(self,backg):
        self.image -= backg.image * 1.0 * self.exposure_time_s / backg.exposure_time_s
        self.mask *= backg.mask 
        self.image *= self.mask 
        
    def parallax_correction(self,**kwargs):
        self.parallax_corrected_position_px = ScaleTools.parallax_correction(size=self.image.shape,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_um=self.pixel_size_um,\
                                                    center_px=self.detector_center_px,\
                                                    thickness_um=self.thickness_um, \
                                                    wavelength_A=self.wavelength_A,\
                                                    absorption_coefficient_invum=self.absorption_coefficient_invum)
    
    def detector_absorption_correction(self,parallax_correction=True,**kwargs):
        if parallax_correction:
            multiply_scaler = ScaleTools.detector_absorption_scaler(size=self.image.shape,\
                                                    center_px=self.detector_center_px,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    parallax_corrected_position_px=self.parallax_corrected_position_px,\
                                                    pixelSize_um=self.pixel_size_um,\
                                                    thickness_um=self.thickness_um,\
                                                    absorption_coefficient_invum=self.absorption_coefficient_invum,\
                                                    wavelength_A=self.wavelength_A)
        else:
            multiply_scaler = ScaleTools.detector_absorption_scaler(size=self.image.shape,\
                                                    center_px=self.detector_center_px,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    parallax_corrected_position_px=None,\
                                                    pixelSize_um=self.pixel_size_um,\
                                                    thickness_um=self.thickness_um,\
                                                    absorption_coefficient_invum=self.absorption_coefficient_invum,\
                                                    wavelength_A=self.wavelength_A)
        self.abs = multiply_scaler
        self.image *= multiply_scaler
    
    def polarization_correction(self,parallax_correction=True,**kwargs):
        if parallax_correction:
            multiply_scaler = ScaleTools.polarization_scaler(size=self.image.shape, \
                                                    parallax_corrected_position=self.parallax_corrected_position_px, \
                                                    polarization=self.polarization_fr, \
                                                    detectorDistance=self.detector_distance_mm, \
                                                    pixelSize=self.pixel_size_mm, \
                                                    center=self.detector_center_px)
        else:
            multiply_scaler = ScaleTools.polarization_scaler(size=self.image.shape, \
                                                    parallax_corrected_position=None, \
                                                    polarization=self.polarization_fr, \
                                                    detectorDistance=self.detector_distance_mm, \
                                                    pixelSize=self.pixel_size_mm, \
                                                    center=self.detector_center_px)
        self.p = multiply_scaler
        self.image *= multiply_scaler 
    
    def solid_angle_correction(self,parallax_correction=True,**kwargs):
        if parallax_correction:
            multiply_scaler = ScaleTools.solid_angle_scaler(size=self.image.shape, \
                                                    parallax_corrected_position=self.parallax_corrected_position_px, \
                                                    detectorDistance=self.detector_distance_mm, \
                                                    pixelSize=self.pixel_size_mm, \
                                                    center=self.detector_center_px)
        else:
            multiply_scaler = ScaleTools.solid_angle_scaler(size=self.image.shape, \
                                                    parallax_corrected_position=None, \
                                                    detectorDistance=self.detector_distance_mm, \
                                                    pixelSize=self.pixel_size_mm, \
                                                    center=self.detector_center_px)
        self.s = multiply_scaler
        self.image *= multiply_scaler
    
    def remove_bragg_peaks(self,replace_by_median=True,window_size_px=11,parallax_correction=True,**kwargs):
#         print replace_by_median,window_size_px,parallax_correction
        parallax_correction_px = None
        if parallax_correction:
            parallax_correction_px = self.parallax_corrected_position_px
            
        peakIdenty = scripts.merging.PeakMask(Amat=self.Amat_invA, _image=self.image, \
                                                    size=None, xvector=None, \
                                                    waveLength=self.wavelength_A, \
                                                    pixelSize=self.pixel_size_mm, \
                                                    center=self.detector_center_px, \
                                                    detectorDistance=self.detector_distance_mm, \
                                                    Phi=self.phi_deg, rotAxis=self.rotAxis,\
                                                    parallax_correction_px=parallax_correction_px)
        self.peakMask = 1-peakIdenty
        if replace_by_median:
            median_image = FilterTools.median_filter(image=self.image,mask=self.mask,\
                                                    window=(window_size_px,window_size_px))
            index = np.where(peakIdenty==1)
            self.image[index] = median_image[index]
            self.image *= self.mask
#             self.median_image = median_image
        else:
            self.image *= self.peakMask
    
    def calculate_radial_profile(self,radius_bin_px=5):
        aveRadius, aveIntens, sumCount = scripts.image.radialProfile(self.image, self.mask, \
                                                    center=self.detector_center_px, \
                                                    vmin=None, vmax=None, rmin=0, rmax=None, \
                                                    stepSize=1, sampling=None, \
                                                    window=radius_bin_px)
#         print aveRadius[:10]
        self.radprofile = aveIntens
    
    def calculate_overall_intensity(self,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4,**kwargs):
#         print radius_rmin_px,radius_rmax_px,radius_rmin_A,radius_rmax_A
        if radius_rmin_A:
            radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
        if radius_rmax_A:
            radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
            
        keepMask = MaskTools.circleMask(self.image.shape, rmin=radius_rmin_px, rmax=radius_rmax_px, \
                                                    center=self.detector_center_px)
        self.overall_intensity = np.sum(self.image * self.mask * keepMask)
    
    def calculate_average_intensity(self,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4,**kwargs):
        if radius_rmin_A:
            radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
        if radius_rmax_A:
            radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
            
        keepMask = MaskTools.circleMask(self.image.shape, rmin=radius_rmin_px, rmax=radius_rmax_px, \
                                                    center=self.detector_center_px)
        
        self.average_intensity = np.sum(self.image * self.mask * keepMask) / np.sum(self.mask * keepMask)
    
    def calculate_water_ring_intensity(self,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=1.0/0.2,radius_rmax_A=1.0/0.55,**kwargs):
        if radius_rmin_A:
            radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
        if radius_rmax_A:
            radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                    detectorDistance_mm=self.detector_distance_mm,\
                                                    pixelSize_mm=self.pixel_size_mm,\
                                                    waveLength_A=self.wavelength_A)
            
        keepMask = MaskTools.circleMask(self.image.shape, rmin=radius_rmin_px, rmax=radius_rmax_px, \
                                                    center=self.detector_center_px)
        
        self.average_water_ring_intensity = np.sum(self.image * self.mask * keepMask) / np.sum(self.mask * keepMask)
        
    def scale_by_dials_bragg(self,reference_obj,**kwargs):
        ## get plot from dials report
        self.per_image_multiply_scale = 1. / scripts.datafile.get_scale_from_dials_expt(self.dials_scaling_angle_deg,\
                                                                   self.dials_scaling_value,\
                                                                   self.phi_deg + reference_obj.start_angle_deg)
        
        reference_multiply_scale = 1. / scripts.datafile.get_scale_from_dials_expt(reference_obj.dials_scaling_angle_deg,\
                                                                   reference_obj.dials_scaling_value,\
                                                                   reference_obj.phi_deg + reference_obj.start_angle_deg)
        
        # x*10, reference*100 -> x/10
        return self.per_image_multiply_scale * 1.0 / reference_multiply_scale

    def scale_by_radial_profile(self,reference_obj,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4,radius_bin_px=5,**kwargs):
        ## minimize the difference between radial profiles
        if radius_rmin_A:
            radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                                  detectorDistance_mm=self.detector_distance_mm,\
                                                                  pixelSize_mm=self.pixel_size_mm,\
                                                                  waveLength_A=self.wavelength_A)
            
        if radius_rmax_A:
            radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                                  detectorDistance_mm=self.detector_distance_mm,\
                                                                  pixelSize_mm=self.pixel_size_mm,\
                                                                  waveLength_A=self.wavelength_A)
        
        rmin_px = int(round(radius_rmin_px))
        rmax_px = int(round(radius_rmax_px))
        
        self.per_image_multiply_scale = np.dot(reference_obj.radprofile[rmin_px:(rmax_px+1)],self.radprofile[rmin_px:(rmax_px+1)]) \
                            / np.dot(self.radprofile[rmin_px:(rmax_px+1)],self.radprofile[rmin_px:(rmax_px+1)])
            
        return self.per_image_multiply_scale
        
    def scale_by_overall_intensity(self,reference_obj,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4,**kwargs):
        ## average intensity in the whole resolution range
        self.calculate_average_intensity(radius_rmin_px=radius_rmin_px,radius_rmax_px=radius_rmax_px,radius_rmin_A=radius_rmin_A,radius_rmax_A=radius_rmax_A)
        self.per_image_multiply_scale = reference_obj.average_intensity * 1.0 / self.average_intensity
        return self.per_image_multiply_scale
        
    def scale_by_water_ring_intensity(self,reference_obj,radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=1.0/0.2,radius_rmax_A=1.0/0.55,**kwargs):
        ## average water ring intensity
        self.calculate_water_ring_intensity(radius_rmin_px=radius_rmin_px,radius_rmax_px=radius_rmax_px,radius_rmin_A=radius_rmin_A,radius_rmax_A=radius_rmax_A)    
        reference_average_intensity = reference_obj.average_water_ring_intensity
        this_average_intensity = self.average_water_ring_intensity
        self.per_image_multiply_scale = reference_average_intensity * 1.0 / this_average_intensity
        return self.per_image_multiply_scale
        
    def expandMask(self,expand_mask_px=1):
        self.mask = MaskTools.expandMask(self.mask, expandSize=(expand_mask_px, expand_mask_px), expandValue=0)
        self.image *= self.mask 
    
    def buildPeakMask(self, bmin=0, bmax=0.25):
        peakIdenty = scripts.merging.PeakMask(Amat=self.Amat_invA, _image=self.image, \
                                              size=None, xvector=None, window=(bmin, bmax), \
                                              waveLength=self.wavelength_A, \
                                              pixelSize=self.pixel_size_mm, \
                                              center=self.detector_center_px, \
                                              detectorDistance=self.detector_distance_mm, \
                                              Phi=self.phi_deg, rotAxis=self.rotAxis)
        peakMask = 1-peakIdenty 
        return peakMask

    def convert2hkl(self):
        hkl = scripts.merging.mapImage2Voxel(image=self.image, size=None, \
                                             Amat=self.Amat_invA, Bmat=None, \
                                             xvector=None, Phi=self.phi_deg, \
                                             waveLength=self.wavelength_A, \
                                             pixelSize=self.pixel_size_mm, \
                                             center=self.detector_center_px, \
                                             detectorDistance=self.detector_distance_mm, \
                                             returnFormat="HKL", voxelSize=1.0, \
                                             rotAxis=self.rotAxis)
        self.hklI = np.zeros((np.prod(self.image.shape),4))
        self.hklI[:,0] = hkl[:,:,0].ravel()
        self.hklI[:,1] = hkl[:,:,1].ravel()
        self.hklI[:,2] = hkl[:,:,2].ravel()
        self.hklI[:,3] = self.image.ravel()
        return True
    
    def mergehkl(self):
        volume, weight = self.merge2volume()
        index = np.where(weight>0)
        volumeScale = volume.copy()
        volumeScale[index] /= weight[index]
        
        self.HKLI = np.zeros((len(index[0]), 4))
        self.HKLIW = np.zeros((len(index[1]), 5))
        
        self.HKLI[:,0] = np.array(index[0]) - 60
        self.HKLI[:,1] = np.array(index[1]) - 60
        self.HKLI[:,2] = np.array(index[2]) - 60
        self.HKLI[:,3] = volumeScale[index]
        
        self.HKLIW[:,0:3] = self.HKLI[:,0:3]
        self.HKLIW[:,3] = volume[index]
        self.HKLIW[:,4] = weight[index]
        return True
    
    def merge2volume(self,keep_bragg_peaks=False,xvector=None,parallax_correction_px=None,\
                        volume_center_vx=60,volume_size_vx=121,oversample=1,window_keep_hkl=(0.25,1)):
        
#         print volume_center_vx, volume_size_vx,oversample
        volume = np.zeros((volume_size_vx,volume_size_vx,volume_size_vx))
        weight = np.zeros((volume_size_vx,volume_size_vx,volume_size_vx))

        volume, weight = scripts.merging.Image2Volume(volume=volume, \
                                                      weight=weight, \
                                                      Amat=self.Amat_invA, \
                                                      Bmat=None, \
                                                      _image=self.image, \
                                                      _mask=self.mask, \
                                                      keepPeak=keep_bragg_peaks, \
                                                      returnFormat="HKL", \
                                                      xvector=xvector, \
                                                      waveLength=self.wavelength_A, \
                                                      pixelSize=self.pixel_size_mm, \
                                                      center=self.detector_center_px, \
                                                      detectorDistance=self.detector_distance_mm, \
                                                      Vcenter=volume_center_vx, \
                                                      Vsize=volume_size_vx, \
                                                      oversample=oversample, \
                                                      Phi=self.phi_deg, \
                                                      rotAxis=self.rotAxis,\
                                                      parallax_correction_px=parallax_correction_px,\
                                                      window_keep_hkl=window_keep_hkl)
        
        self.volume = volume
        self.weight = weight
        return volume, weight

    def convert2reciprocal(self, image=None):
        if image is None:
            return None
        xyz = scripts.merging.mapPixel2RealXYZ(size=image.shape, \
                                               center=self.detector_center_px, \
                                               pixelSize=self.pixel_size_mm, \
                                               detectorDistance=self.detector_distance_mm)
        rep = scripts.merging.mapRealXYZ2Reciprocal(xyz=xyz, waveLength=self.wavelength_A)
        endaxis = len(rep.shape)-1
        return np.sqrt(np.sum(rep**2, axis=endaxis))
        
        

## A cluster a diffraction patterns
class MergeAgent:
    def __init__(self):
        self.cluster = {}
        self.numImage = 0
        self.startnum = 0 
        
    def todict(self):
        return self.__dict__
    
    def fromdict(self, entry):
        self.__dict__.update(entry) 
        
    def readfile(self, filename=None):
        if filename is None:
            return
        psvm = scripts.datafile.loadfile(filename)
        return psvm
    
    def addfile(self,data):
        if len(self.cluster) != 0:
            maxIdx = int(max(self.cluster))
        else:
            maxIdx = self.startnum-1
            
        nextIdx = "%.5d"%(maxIdx+1)
        self.cluster[nextIdx] = data
        self.numImage = len(self.cluster)
        return True
    
    
    def pushparams(self, params):
        for key in self.cluster:
            self.cluster[key].update(params)
        return True
    
    
    def merge_by_averaging(self,keep_bragg_peaks=False,xvector=None,\
                    volume_center_vx=60,volume_size_vx=121,oversample=1,window_keep_hkl=(0.25,1),**kwargs):
#         print volume_center_vx, volume_size_vx,oversample
        volume = np.zeros((volume_size_vx,volume_size_vx,volume_size_vx))
        weight = np.zeros((volume_size_vx,volume_size_vx,volume_size_vx))
        
        for each in sorted(self.cluster):
            filename = self.cluster[each]["image_file"]
#             print "start image file: ", filename
            imageAgent = ImageAgent()
            imageAgent.loadImage(filename,fileType="h5")
            imageAgent.scale = self.cluster[each]["per_image_multiply_scale"]
            imageAgent.image *= imageAgent.scale
            ################
            parallax_correction_px = None
            if hasattr(imageAgent,"parallax_corrected_position_px"):
                parallax_correction_px = imageAgent.parallax_corrected_position_px
            ################
            v,w = imageAgent.merge2volume(keep_bragg_peaks=keep_bragg_peaks,\
                                          xvector=xvector,\
                                          parallax_correction_px=parallax_correction_px,\
                                          volume_center_vx=volume_center_vx,\
                                          volume_size_vx=volume_size_vx,\
                                          oversample=oversample,\
                                          window_keep_hkl=window_keep_hkl)
            volume += v
            weight += w
            imageAgent = None
            print "#### Merged image file: ", filename
            
        self.volume = volume
        self.weight = weight
        return volume,weight
    