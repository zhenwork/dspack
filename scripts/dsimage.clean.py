import os,sys
import numpy as np
from numba import jit

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname",help="input data.dsdata", default="./data.dsdata", type=str)
parser.add_argument("-fsave","--fsave",help="save new dsdata file", default=None, type=str)

parser.add_argument("-read_dname","--read_dname",help="read dname", default="image_file", type=str)
parser.add_argument("-save_dname","--save_dname",help="save dname", default=None, type=str)

parser.add_argument("-save_dir","--save_dir",help="save folder", default="./datadir", type=str)

parser.add_argument("--apply_detector_mask",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--subtract_background",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--remove_bad_pixels",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--parallax_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--polarization_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--solid_angle_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--detector_absorption_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--remove_bragg_peaks",help="cases: None, [], [1,2,3]",default=None,nargs="*")
args = parser.parse_args()

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.manager
import scripts.fsystem
from scripts.mpidata import *

#### get process params
args.apply_detector_mask = scripts.utils.get_process_params(args.apply_detector_mask)
args.remove_bad_pixels = scripts.utils.get_process_params(args.remove_bad_pixels)
args.subtract_background = scripts.utils.get_process_params(args.subtract_background)
args.parallax_correction = scripts.utils.get_process_params(args.parallax_correction)
args.polarization_correction = scripts.utils.get_process_params(args.polarization_correction)
args.solid_angle_correction = scripts.utils.get_process_params(args.solid_angle_correction)
args.detector_absorption_correction = scripts.utils.get_process_params(args.detector_absorption_correction)
args.remove_bragg_peaks = scripts.utils.get_process_params(args.remove_bragg_peaks)


#### create folders
args.fsave = args.fsave or args.fname
args.save_dname = args.save_dname or args.read_dname
unique_time_stamp = ""
if comm_rank == 0:
    if not os.path.isdir(os.path.realpath(args.save_dir)):
        os.makedirs(args.save_dir)
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)
    unique_time_stamp = scripts.utils.get_time()
unique_time_stamp = comm.bcast(unique_time_stamp, root=0)
comm.Barrier()


#### load data from file
import_data = scripts.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname, \
                                                                     "backg_file", \
                                                                     "gxparms_file",\
                                                                     "detector_mask_file",\
                                                                     "extra_params"])
num_images = len(import_data.get(args.read_dname))
print("#### Rank %3d/%3d started"%(comm_rank,comm_size))

#### process the pipeline
for image_idx in range(num_images):
    if image_idx%comm_size != comm_rank:
        continue
    
    image_obj = scripts.manager.ImageAgent()
    image_obj.loadImage(import_data.get(args.read_dname)[image_idx])
    image_obj.loadImage(import_data.get("gxparms_file"))
    image_obj.detector_mask_file = import_data.get("detector_mask_file")
    if image_obj.phi_deg is None:
        image_obj.phi_deg = image_obj.angle_step_deg * 1.0 * image_idx
    image_obj.loadImage(import_data.get("extra_params"))
    
    if args.subtract_background.status:
        backg_obj = scripts.manager.ImageAgent()
        backg_obj.loadImage(import_data.get("backg_file")[image_idx])
        backg_obj.loadImage(import_data.get("gxparms_file"))
        backg_obj.detector_mask_file = import_data.get("detector_mask_file")
        if backg_obj.phi_deg is None:
            backg_obj.phi_deg = backg_obj.angle_step_deg * 1.0 * image_idx
        backg_obj.loadImage(import_data.get("extra_params"))

    # apply detector mask
    # default: radius_rmin_px=40,radius_rmax_px=None,value_vmin=0,value_vmax=10000
    if args.apply_detector_mask.status:
        image_obj.apply_detector_mask(**args.apply_detector_mask.params)
        if args.subtract_background.status:
            backg_obj.apply_detector_mask(**args.apply_detector_mask.params)

    # remove bad pixels
    # default: algorithm=2,sigma=5,window_size_px=11
    if args.remove_bad_pixels.status:
        image_obj.remove_bad_pixels(**args.remove_bad_pixels.params)
        if args.subtract_background.status:
            backg_obj.remove_bad_pixels(**args.remove_bad_pixels.params)

    # subtract background
    # default: {}
    if args.subtract_background.status:
        image_obj.subtract_background(backg=backg_obj,**args.subtract_background.params)

    # parallax correction 
    # default: {}
    if args.parallax_correction.status:
        image_obj.parallax_correction(**args.parallax_correction.params)

    # polarization correction 
    # default: parallax_correction=True
    if args.polarization_correction.status:
        image_obj.polarization_correction(parallax_correction=args.parallax_correction.status,**args.polarization_correction.params)

    # solid angle correction 
    # default: parallax_correction=True
    if args.solid_angle_correction.status:
        image_obj.solid_angle_correction(parallax_correction=args.parallax_correction.status,**args.solid_angle_correction.params)

    # detector absorption correction
    # default: parallax_correction=True,
    if args.detector_absorption_correction.status:
        image_obj.detector_absorption_correction(parallax_correction=args.parallax_correction.status,**args.detector_absorption_correction.params)

    # remove bragg peaks
    # default: replace_by_median=True,window_size_px=11,parallax_correction=True,
    if args.remove_bragg_peaks.status:
        image_obj.remove_bragg_peaks(parallax_correction=args.parallax_correction.status,**args.remove_bragg_peaks.params)
    
    fsave_image = os.path.join(args.save_dir,"%s_%s_%.5d.h5"%(args.save_dname,unique_time_stamp,image_idx))
    scripts.fsystem.PVmanager.modify(image_obj.__dict__,fsave_image)
    print("#### Rank %3d/%3d has cleaned image: %5d"%(comm_rank,comm_size,image_idx))

comm.Barrier()

if comm_rank==0:
    processed_file = [os.path.realpath(os.path.join(args.save_dir,"%s_%s_%.5d.h5"%(args.save_dname,unique_time_stamp,image_idx))) \
                                          for image_idx in range(num_images)]
    data_save = {args.save_dname:processed_file}
    scripts.fsystem.PVmanager.modify(data_save, args.fsave)

    print("#### Image Clean Done")