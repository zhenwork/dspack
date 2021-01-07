import os,sys
import numpy as np
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.image
import scripts.manager
import scripts.fsystem
import scripts.datafile 
from scripts.mpidata import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname",help="input files", default="./data.diffuse", type=str) 
parser.add_argument("-fsave","--fsave",help="save new files", default=None, type=str) 
parser.add_argument("-read_dname","--read_dname",help="save new files", default="image_file", type=str) 
parser.add_argument("-save_dname","--save_dname",help="save new files", default="scale_factor", type=str)

parser.add_argument("--scale_by_radial_profile",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_overall_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_water_ring_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_bragg_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_nothing",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
args = parser.parse_args() 


#### create folders
args.fsave = args.fsave or args.fname
unique_time_stamp = "random"
if comm_rank == 0:
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)
    unique_time_stamp = scripts.utils.get_time()
unique_time_stamp = comm.bcast(unique_time_stamp, root=0)
comm.Barrier()

#### load data from file
import_data = scripts.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname,"dials_report_file","extra_params"])
num_images = len(import_data.get(args.read_dname))


#### get process params
args.scale_by_nothing = scripts.utils.get_process_params(args.scale_by_nothing)
args.scale_by_radial_profile = scripts.utils.get_process_params(args.scale_by_radial_profile)
args.scale_by_bragg_intensity = scripts.utils.get_process_params(args.scale_by_bragg_intensity)
args.scale_by_overall_intensity = scripts.utils.get_process_params(args.scale_by_overall_intensity)
args.scale_by_water_ring_intensity = scripts.utils.get_process_params(args.scale_by_water_ring_intensity)

radial_profile_stack = None
per_image_multiply_scale = np.zeros(num_images)
reference_obj = None

#### process the pipeline
for image_idx in range(num_images):
    if image_idx%comm_size != comm_rank:
        continue

    file_name = import_data.get(args.read_dname)[image_idx]
    image_obj = scripts.manager.ImageAgent()
    image_obj.loadImage(file_name)
    image_obj.loadImage(import_data.get("dials_report_file"))
    image_obj.loadImage(import_data.get("extra_params"))
    
    image_obj.calculate_radial_profile(**args.scale_by_radial_profile.params)
    if radial_profile_stack is None:
        radial_profile_stack = np.zeros((num_images, len(image_obj.radprofile)))
    radial_profile_stack[image_idx] = image_obj.radprofile
    
    # calculate bragg intensity 
    # default: {}
    if args.scale_by_bragg_intensity.status:
        if not reference_obj:
            reference_idx = args.scale_by_bragg_intensity.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx])
            reference_obj.loadImage(import_data.get("dials_report_file"))
            reference_obj.loadImage(import_data.get("extra_params"))
        per_image_multiply_scale[image_idx] = image_obj.scale_by_dials_bragg(reference_obj=reference_obj,\
                                                                             **args.scale_by_bragg_intensity.params)
    # calculate radial profile intensity 
    # default: radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4,radius_bin_px=5
    elif args.scale_by_radial_profile.status:
        if not reference_obj:
            reference_idx = args.scale_by_radial_profile.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx]) 
            reference_obj.loadImage(import_data.get("extra_params"))
            reference_obj.calculate_radial_profile(**args.scale_by_radial_profile.params)
        per_image_multiply_scale[image_idx] = image_obj.scale_by_radial_profile(reference_obj=reference_obj,\
                                                                            **args.scale_by_radial_profile.params)
        

    # calculate overall intensity 
    # default: radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=50,radius_rmax_A=1.4
    elif args.scale_by_overall_intensity.status:
        if not reference_obj:
            reference_idx = args.scale_by_overall_intensity.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx]) 
            reference_obj.loadImage(import_data.get("extra_params"))
            reference_obj.calculate_average_intensity(**args.scale_by_overall_intensity.params)
        per_image_multiply_scale[image_idx] = image_obj.scale_by_overall_intensity(reference_obj=reference_obj,\
                                                                            **args.scale_by_overall_intensity.params)

    # calculate water ring intensity 
    # default: radius_rmin_px=None,radius_rmax_px=None,radius_rmin_A=1.0/0.2,radius_rmax_A=1.0/0.55
    elif args.scale_by_water_ring_intensity.status:
        if not reference_obj:
            reference_idx = args.scale_by_water_ring_intensity.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx]) 
            reference_obj.loadImage(import_data.get("extra_params"))    
            reference_obj.calculate_water_ring_intensity(**args.scale_by_water_ring_intensity.params)
        per_image_multiply_scale[image_idx] = image_obj.scale_by_water_ring_intensity(reference_obj=reference_obj,\
                                                                            **args.scale_by_water_ring_intensity.params)
    elif args.scale_by_nothing.status:
        per_image_multiply_scale[image_idx] = 1.0
    else:
        raise Exception("!!! Not A Valid Scale Factor")
    
    print("#### Rank %3d/%3d has scaled image: %5d"%(comm_rank,comm_size,image_idx))
    image_obj = None
reference_obj = None

if comm_rank != 0:
    md=mpidata()
    md.addarray('per_image_multiply_scale', per_image_multiply_scale) 
    md.addarray('radial_profile_stack', radial_profile_stack) 
    md.small.rank = comm_rank 
    md.send()
    md = None
else:
    for nrank in range(comm_size-1):
        md=mpidata()
        md.recv()
        per_image_multiply_scale += md.per_image_multiply_scale 
        radial_profile_stack += md.radial_profile_stack
        recvRank = md.small.rank
        md = None
        print '#### Rank 0 received file from Rank ' + str(recvRank).rjust(2) 
comm.Barrier()

if comm_rank==0:
    scale_dir = os.path.dirname(file_name)
    scale_file = os.path.join(scale_dir,"%s_%s.npy"%(args.save_dname,unique_time_stamp))
    np.save(scale_file, per_image_multiply_scale)
    
    profile_file = os.path.join(scale_dir,"%s_%s.npy"%("radial_profile_stack",unique_time_stamp))
    np.save(profile_file, radial_profile_stack)

    data_save = {args.save_dname:scale_file, "radial_profile_stack":profile_file}
    scripts.fsystem.PVmanager.modify(data_save, args.fsave)
    print("#### Image Scale Done")