import os,sys
import numpy as np
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.manager
import scripts.fsystem
import scripts.datafile 
import scripts.merging
from scripts.mpidata import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default="./data.dsdata", type=str)
parser.add_argument("-fsave","--fsave", help="save folder", default=None, type=str)
parser.add_argument("-read_dname","--read_dname", help="merge dname ", default="image_file", type=str)
parser.add_argument("-save_dname","--save_dname", help="save folder", default="merge_volume", type=str)

parser.add_argument("--per_image_multiply_scale",default="scale_factor",type=str)
parser.add_argument("--select_image_idx", help="select idx of images", default=None, type=str)
parser.add_argument("--merge_to_volume",help="cases: None, [], [1,2,3]",default=None,nargs="*")
args = parser.parse_args()


#### 
args.fsave = args.fsave or args.fname
unique_time_stamp = "random"
if comm_rank == 0:
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)
    unique_time_stamp = scripts.utils.get_time()
unique_time_stamp = comm.bcast(unique_time_stamp, root=0)
comm.Barrier()


#### load data from file
import_data = scripts.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname,args.per_image_multiply_scale])
num_images = len(import_data.get(args.read_dname))

per_image_multiply_scale = np.load(import_data.get(args.per_image_multiply_scale))
if comm_rank > 0:
    select_image_idx = scripts.utils.get_array_list(args.select_image_idx) or range(num_images)
    assign_image_idx = select_image_idx[(comm_rank-1)::(comm_size-1)]
    print "#### Rank %d/%d processes [%d,%d,...,%d,%d]"%(comm_rank,comm_size,assign_image_idx[0],assign_image_idx[1],\
                                                        assign_image_idx[-2],assign_image_idx[-1])


#### run image merging
args.merge_to_volume = scripts.utils.get_process_params(args.merge_to_volume)
if args.merge_to_volume.status: 
    volume_size_px = args.merge_to_volume.params.get("volume_size_px") or 121
    volume_center_px = args.merge_to_volume.params.get("volume_center_px") or 60

if comm_rank != 0:
    mergeAgent = scripts.manager.MergeAgent()
    for image_idx in assign_image_idx:
        file_name = import_data.get(args.read_dname)[image_idx]
        multiply_scaler = per_image_multiply_scale[image_idx]
#         print "#### Rank %3d/%3d adds file: %s"%(comm_rank, comm_size, file_name) 
        mergeAgent.addfile({"image_file":file_name,"per_image_multiply_scale":multiply_scaler})
        
    ## merge_to_volume
    ## default: keep_bragg_peaks=False,xvector=None,\
    ## .. default: volume_center_vx=60,volume_size_vx=121,oversample=1,window_keep_hkl=(0.25,1)
    if args.merge_to_volume.status:
        volume,weight = mergeAgent.merge_by_averaging(**args.merge_to_volume.params)
    
    md=mpidata()
    md.addarray('volume', mergeAgent.volume)
    md.addarray('weight', mergeAgent.weight)
    md.small.rank = comm_rank
    md.small.num_merge = len(assign_image_idx)
    md.send()
    md = None
    
else:
    volume = 0.
    weight = 0.
    for nrank in range(comm_size-1):
        md=mpidata()
        md.recv()
        volume += md.volume
        weight += md.weight
        recvRank = md.small.rank
        md = None
        print '#### Rank 0 received file from Rank ' + str(recvRank).rjust(2) 
    
    index = np.where(weight>=4)
    volume[index] /= weight[index]
    index = np.where(weight<4)
    volume[index] = -1024
    weight[index] = 0

comm.Barrier()

if comm_rank == 0:
    file_name = import_data.get(args.read_dname)[0]
    volume_dir = os.path.realpath(os.path.dirname(file_name))
    volume_file = os.path.join(volume_dir,"%s_%s.h5"%(args.save_dname,unique_time_stamp))
    data_save = {args.save_dname:volume_file}

    scripts.fsystem.PVmanager.modify(data_save, args.fsave)
    
    ## write volume into files
    psvm = scripts.datafile.reader(file_name,"h5")
    Amat_invA = psvm["Amat_invA"]
    Bmat_invA = psvm["Bmat_invA"]
    
    scripts.fsystem.H5manager.modify(volume_file, 'volume', volume, chunks=(1,volume_size_px,volume_size_px), \
                                     compression="gzip", compression_opts=7)
    scripts.fsystem.H5manager.modify(volume_file, 'weight', weight, chunks=(1,volume_size_px,volume_size_px), \
                                     compression="gzip", compression_opts=7)
    scripts.fsystem.H5manager.modify(volume_file, 'Amat_invA', Amat_invA)
    scripts.fsystem.H5manager.modify(volume_file, 'Bmat_invA', Bmat_invA)
    scripts.fsystem.H5manager.modify(volume_file, 'Smat_invA', Bmat_invA/scripts.utils.length(Bmat_invA[:,1]))
    
    scripts.fsystem.H5manager.modify(volume_file, 'oversample', args.merge_to_volume.params.get("oversample") or 1)