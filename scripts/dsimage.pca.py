import h5py
import os,sys
import numpy as np
from numba import jit
from sklearn.decomposition import PCA

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.merging
import scripts.fsystem
import scripts.manager
from scripts.mpidata import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="input files", default="./data.diffuse", type=str) 
parser.add_argument("--fsave",help="save new files", default=None, type=str) 
parser.add_argument("--read_dname",help="read dname", default="image_file", type=str) 
parser.add_argument("--save_dname",help="save dname", default=None, type=str) 
parser.add_argument("--per_image_multiply_scale",default="scale_factor",type=str)
parser.add_argument("--radial_profile_stack",default="radial_profile_stack",type=str)
parser.add_argument("--radial_pca_subtraction",help="",default=None,nargs="*") 
args = parser.parse_args() 

#### make files
args.fsave = args.fsave or args.fname
args.save_dname = args.save_dname or args.read_dname
unique_time_stamp = "random"
if comm_rank == 0:
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)
    unique_time_stamp = scripts.utils.get_time()
unique_time_stamp = comm.bcast(unique_time_stamp, root=0)
comm.Barrier()

#### load data from file
import_data = scripts.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname,\
                                                                     args.per_image_multiply_scale, \
                                                                     args.radial_profile_stack ])
num_images = len(import_data.get(args.read_dname))

radial_profile_stack = np.load(import_data.get(args.radial_profile_stack))
per_image_multiply_scale = np.load(import_data.get(args.per_image_multiply_scale))

## get action params
args.radial_pca_subtraction = scripts.utils.get_process_params(args.radial_pca_subtraction)


## get required params
fname = import_data.get(args.read_dname)[0]
image_obj = scripts.fsystem.PVmanager.reader(fname)
image_size_px = image_obj["image"].shape
pixel_size_mm = image_obj["pixel_size_mm"]
wavelength_A = image_obj["wavelength_A"] 
detector_center_px = image_obj["detector_center_px"]
detector_distance_mm = image_obj["detector_distance_mm"]
image_obj = None


## load the radial profile
def process_one_single_pca(radial_profile_stack):
    pca = PCA()
    pca.fit(radial_profile_stack)
    fitted_data = pca.transform(radial_profile_stack)
    
    fitted_data[:,1:] = 0 
    backg_profile_stack = fitted_data.dot(pca.components_) 
    backg_profile_stack[radial_profile_stack==0] = 0
    return backg_profile_stack


def process_multiple_pca(radial_profile_stack,num_pca):
    pca = PCA()
    pca.fit(radial_profile_stack)
    fitted_data = pca.transform(radial_profile_stack)
    
    fitted_data[:, num_pca:] = 0 
    backg_profile_stack = fitted_data.dot(pca.components_) 
    backg_profile_stack[radial_profile_stack==0] = 0
    return backg_profile_stack


def calculate_pca_background(radial_profile_stack=None,radius_rmin_A=50,radius_rmax_A=1.4,\
                             radius_rmin_px=None,radius_rmax_px=None,num_pca=3,\
                             wavelength_A=None,pixel_size_mm=None,\
                             detector_distance_mm=None):
    if radius_rmin_A:
        radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                detectorDistance_mm=detector_distance_mm,\
                                                pixelSize_mm=pixel_size_mm,\
                                                waveLength_A=wavelength_A)
    if radius_rmax_A:
        radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                detectorDistance_mm=detector_distance_mm,\
                                                pixelSize_mm=pixel_size_mm,\
                                                waveLength_A=wavelength_A)
    
    rmin_px = int(round(max(radius_rmin_px, 0)))
    rmax_px = int(round(min(radius_rmax_px, radial_profile_stack.shape[1]-1)))
    
    rmin_px_has_value = np.amin(np.argmax(radial_profile_stack>0,axis=1))
    rmax_px_has_value = np.amax(radial_profile_stack.shape[1] - 1 - np.argmax(radial_profile_stack[:,::-1]>0,axis=1)[::-1])
    rmin_px = max(rmin_px, rmin_px_has_value)
    rmax_px = min(rmax_px, rmax_px_has_value)
    
    radial_profile_stack_cut = radial_profile_stack[:,rmin_px:rmax_px+1].copy()
    backg_profile_stack_cut = np.zeros(radial_profile_stack_cut.shape)
    for idx_pca in range(num_pca):
        single_backg_profile_stack_cut = process_one_single_pca(radial_profile_stack_cut)
        backg_profile_stack_cut += single_backg_profile_stack_cut
        radial_profile_stack_cut -= single_backg_profile_stack_cut
    backg_profile_stack = np.zeros(radial_profile_stack.shape)
    backg_profile_stack[:,rmin_px:rmax_px+1] = backg_profile_stack_cut
    return backg_profile_stack,rmin_px,rmax_px

def calculate_multi_pca_background(radial_profile_stack=None,radius_rmin_A=50,radius_rmax_A=1.4,\
                             radius_rmin_px=None,radius_rmax_px=None,num_pca=3,\
                             wavelength_A=None,pixel_size_mm=None,\
                             detector_distance_mm=None):
    if radius_rmin_A:
        radius_rmin_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmin_A,\
                                                detectorDistance_mm=detector_distance_mm,\
                                                pixelSize_mm=pixel_size_mm,\
                                                waveLength_A=wavelength_A)
    if radius_rmax_A:
        radius_rmax_px = scripts.merging.resolution_to_radius(resolution_A=radius_rmax_A,\
                                                detectorDistance_mm=detector_distance_mm,\
                                                pixelSize_mm=pixel_size_mm,\
                                                waveLength_A=wavelength_A)
    
    rmin_px = int(round(max(radius_rmin_px, 0)))
    rmax_px = int(round(min(radius_rmax_px, radial_profile_stack.shape[1]-1)))
    
    rmin_px_has_value = np.amin(np.argmax(radial_profile_stack>0,axis=1))
    rmax_px_has_value = np.amax(radial_profile_stack.shape[1] - 1 - np.argmax(radial_profile_stack[:,::-1]>0,axis=1)[::-1])
    rmin_px = max(rmin_px, rmin_px_has_value)
    rmax_px = min(rmax_px, rmax_px_has_value)
    
    radial_profile_stack_cut = radial_profile_stack[:,rmin_px:rmax_px+1].copy()
    backg_profile_stack_cut = np.zeros(radial_profile_stack_cut.shape)
    """
    for idx_pca in range(num_pca):
        single_backg_profile_stack_cut = process_one_single_pca(radial_profile_stack_cut)
        backg_profile_stack_cut += single_backg_profile_stack_cut
        radial_profile_stack_cut -= single_backg_profile_stack_cut
    """
    multiple_backg_profile_stack_cut = process_multiple_pca(radial_profile_stack_cut,num_pca)
    backg_profile_stack_cut += multiple_backg_profile_stack_cut
    
    backg_profile_stack = np.zeros(radial_profile_stack.shape)
    backg_profile_stack[:,rmin_px:rmax_px+1] = backg_profile_stack_cut
    return backg_profile_stack,rmin_px,rmax_px


if comm_rank == 0:
    scaled_radial_profile_stack = np.zeros(radial_profile_stack.shape)
    for image_idx in range(num_images):
        scaled_radial_profile_stack[image_idx] = radial_profile_stack[image_idx] * per_image_multiply_scale[image_idx]
        
    ## radial profile variance removal 
    # default: radius_rmin_A=50,radius_rmax_A=1.4,radius_rmin_px=None,radius_rmax_px=None,num_pca=3
    if args.radial_pca_subtraction.status:
        scaled_backg_profile_stack,rmin_px,rmax_px = calculate_multi_pca_background(radial_profile_stack=scaled_radial_profile_stack,\
                                                             wavelength_A=wavelength_A,pixel_size_mm=pixel_size_mm,\
                                                             detector_distance_mm=detector_distance_mm,\
                                                             **args.radial_pca_subtraction.params)
else:
    scaled_backg_profile_stack = None
    rmin_px = 1000
    rmax_px = 10
print "#### loaded the radial profile files"

## broadcast the radial profile
rmin_px = comm.bcast(rmin_px, root=0)  
rmax_px = comm.bcast(rmax_px, root=0)  
scaled_backg_profile_stack = comm.bcast(scaled_backg_profile_stack, root=0)  
comm.Barrier()

#### process the pipeline
x = np.arange(image_size_px[0]) - detector_center_px[0]
y = np.arange(image_size_px[1]) - detector_center_px[1]
xaxis,yaxis = np.meshgrid(x,y,indexing="ij")
image_radius_px = np.around(np.sqrt(xaxis**2+yaxis**2)).astype(int)
index_radius_cut = np.where((image_radius_px<rmin_px)|(image_radius_px>rmax_px))

for image_idx in range(num_images):
    if image_idx%comm_size != comm_rank:
        continue

    file_name = import_data.get(args.read_dname)[image_idx]
    
    image_obj = scripts.manager.ImageAgent()
    image_obj.loadImage(file_name)
    
    backg_profile = scaled_backg_profile_stack[image_idx] / per_image_multiply_scale[image_idx]
    backg_pattern = backg_profile[image_radius_px]
    image_obj.image -= backg_pattern
    image_obj.image[index_radius_cut] = 0
    image_obj.mask[index_radius_cut] = 0
    
    pca_corrected_dir = os.path.realpath(os.path.dirname(file_name))
    pca_corrected_file = os.path.join(pca_corrected_dir,"%s_%s_%.5d.h5"%(args.save_dname,unique_time_stamp,image_idx))
    scripts.fsystem.PVmanager.modify(image_obj.__dict__,pca_corrected_file)
    print "#### Rank %3d/%3d has normalized image: %5d"%(comm_rank,comm_size,image_idx)

comm.Barrier()


if comm_rank==0:
    scale_dir = os.path.dirname(file_name)
    
    before_scaled_radial_profile_stack = scaled_radial_profile_stack.copy()
    pca_scaled_radial_profile_stack = scaled_backg_profile_stack.copy()
    after_scaled_radial_profile_stack = before_scaled_radial_profile_stack - pca_scaled_radial_profile_stack
    after_scaled_radial_profile_stack[:,:rmin_px] = 0
    after_scaled_radial_profile_stack[:,(rmax_px+1):] = 0
    ################################################
    pca_unscaled_radial_profile_stack = np.zeros(pca_scaled_radial_profile_stack.shape)
    after_unscaled_radial_profile_stack = np.zeros(after_scaled_radial_profile_stack.shape)
    for image_idx in range(num_images):
        after_unscaled_radial_profile_stack[image_idx] = after_scaled_radial_profile_stack[image_idx] / per_image_multiply_scale[image_idx]
        pca_unscaled_radial_profile_stack[image_idx] = pca_scaled_radial_profile_stack[image_idx] / per_image_multiply_scale[image_idx]
    ################################################
    
    pca_backg_stack_unscaled_file = os.path.join(scale_dir,"%s_%s.npy"%("pca_backg_stack",unique_time_stamp))
    np.save(pca_backg_stack_unscaled_file, pca_unscaled_radial_profile_stack)
    
    radial_profile_stack_file = os.path.join(scale_dir,"%s_%s.npy"%("radial_profile_stack",unique_time_stamp))
    np.save(radial_profile_stack_file, after_unscaled_radial_profile_stack)
    
    pca_corrected_file = [os.path.realpath(os.path.join(pca_corrected_dir,"%s_%s_%.5d.h5"%(args.save_dname,unique_time_stamp,image_idx))) \
                         for image_idx in range(num_images)]
    
    data_save = { args.save_dname:pca_corrected_file, \
                 "pca_profile_stack":pca_backg_stack_unscaled_file,\
                 "radial_profile_stack":radial_profile_stack_file}
    
    scripts.fsystem.PVmanager.modify(data_save, args.fsave)