import os,sys
import numpy as np
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.fsystem
import scripts.datafile 
import scripts.volume 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default="./data.dsdata", type=str)
parser.add_argument("-read_dname","--read_dname", help="merge dname", default="merge_volume", type=str) 

parser.add_argument("--laue_symmetrization",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--background_subtraction",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--coordinate_conversion",help="cases: None, [], [1,2,3]",default=None,nargs="*")
parser.add_argument("--friedel_symmetrization",help="cases: None, [], [1,2,3]",default=None,nargs="*")
args = parser.parse_args()


#### 
args.laue_symmetrization = scripts.utils.get_process_params(args.laue_symmetrization)
args.background_subtraction = scripts.utils.get_process_params(args.background_subtraction)
args.coordinate_conversion = scripts.utils.get_process_params(args.coordinate_conversion)
args.friedel_symmetrization = scripts.utils.get_process_params(args.friedel_symmetrization) 


### load files
import_data = scripts.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname])
volume_data = scripts.fsystem.PVmanager.reader(import_data.get(args.read_dname),\
                            keep_keys=["volume","weight","Amat_invA","Bmat_invA"])

volume = volume_data["volume"]
weight = volume_data["weight"]
Amat_invA = volume_data["Amat_invA"]
Bmat_invA = volume_data["Bmat_invA"]
vMask = (weight>=4).astype(int)
index = np.where(weight<4)
volume[index] = 0
weight[index] = 0
vMask[index] = 0

a = Bmat_invA[:,0].copy()
b = Bmat_invA[:,1].copy()
c = Bmat_invA[:,2].copy()

astar = a / scripts.utils.length(b)
bstar = b / scripts.utils.length(b)
cstar = c / scripts.utils.length(b)



data_save = {}
volume_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume,\
                                           astar=astar,bstar=bstar,cstar=cstar,\
                                           volume_mask=vMask,\
                                           threshold_keep=(0.001,1000), \
                                           **args.coordinate_conversion.params)
data_save["volume_xyz"] = volume_xyz
        
        
if args.laue_symmetrization.status:
    print "#### Generating volume_laue_sym"
    volume_laue_sym, weight_laue_sym = scripts.volume.volume_symmetrize(volume_3d=volume,\
                                                                        volume_mask_3d=vMask, \
                                                                        threshold_keep=(0,1000),\
                                                                        **args.laue_symmetrization.params)
    volume_laue_sym[weight_laue_sym<=0] = -1024
    data_save["volume_laue_sym"] = volume_laue_sym
    data_save["weight_laue_sym"] = weight_laue_sym
    
    volume_laue_sym_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_laue_sym,\
                                           astar=astar,bstar=bstar,cstar=cstar,\
                                           volume_mask=(weight_laue_sym>0).astype(int),\
                                           threshold_keep=(0.001,1000), \
                                           **args.coordinate_conversion.params)
    data_save["volume_laue_sym_xyz"] = volume_laue_sym_xyz

    
    if args.background_subtraction.status:
        print "#### Generating volume_laue_sym_backg_sub"
        volume_laue_sym_backg = scripts.volume.volume_radial_profile(volume_3d=volume_laue_sym, \
                                                          volume_mask_3d=(weight_laue_sym>0).astype(int), \
                                                          basis_by_column=Bmat_invA/scripts.utils.length(Bmat_invA[:,1]),\
                                                          threshold_keep=(0,1000), \
                                                          **args.background_subtraction.params)
        
        volume_laue_sym_backg_sub = volume_laue_sym - volume_laue_sym_backg
        volume_laue_sym_backg_sub[weight_laue_sym<=0] = -1024
        data_save["volume_laue_sym_backg_sub"] =  volume_laue_sym_backg_sub
        
        if args.coordinate_conversion.status:
            print "#### Generating volume_laue_sym_backg_sub_xyz"
            volume_laue_sym_backg_sub_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_laue_sym_backg_sub,\
                                                       astar=astar,bstar=bstar,cstar=cstar,\
                                                       volume_mask=(weight_laue_sym>0).astype(int),\
                                                       threshold_keep=(-100,1000), \
                                                       **args.coordinate_conversion.params)
            
            data_save["volume_laue_sym_backg_sub_xyz"] = volume_laue_sym_backg_sub_xyz
            
if args.background_subtraction.status:
    print "#### Generating volume_backg_sub"
    volume_backg = scripts.volume.volume_radial_profile(volume_3d=volume, \
                                                      volume_mask_3d=vMask.copy(), \
                                                      basis_by_column=Bmat_invA/scripts.utils.length(Bmat_invA[:,1]),\
                                                      threshold_keep=(0,1000), \
                                                      **args.background_subtraction.params)
    
    volume_backg_sub = volume - volume_backg
    volume_backg_sub[vMask<=0] = -1024
    data_save["volume_backg_sub"] = volume_backg_sub
    
    volume_backg_sub_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_backg_sub,\
                                                       astar=astar,bstar=bstar,cstar=cstar,\
                                                       volume_mask=vMask,\
                                                       threshold_keep=(-100,1000), \
                                                       **args.coordinate_conversion.params) 
            
    data_save["volume_backg_sub_xyz"] = volume_backg_sub_xyz
            
    if args.laue_symmetrization.status:
        print "#### Generating volume_backg_sub_laue_sym"
        volume_backg_sub_laue_sym,weight_laue_sym = scripts.volume.volume_symmetrize(volume_3d=volume_backg_sub,\
                                                                        volume_mask_3d=vMask, \
                                                                        threshold_keep=(-100,1000),\
                                                                        **args.laue_symmetrization.params)
        
        volume_backg_sub_laue_sym[weight_laue_sym==0] = -1024
        data_save["volume_backg_sub_laue_sym"] = volume_backg_sub_laue_sym
        data_save["weight_laue_sym"] = weight_laue_sym
        
        if args.coordinate_conversion.status:
            print "#### Generating volume_backg_sub_laue_sym_xyz"
            volume_backg_sub_laue_sym_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_backg_sub_laue_sym,\
                                                       astar=astar,bstar=bstar,cstar=cstar,\
                                                       volume_mask=(weight_laue_sym>0).astype(int),\
                                                       threshold_keep=(-100,1000), \
                                                       **args.coordinate_conversion.params) 
            
            data_save["volume_backg_sub_laue_sym_xyz"] = volume_backg_sub_laue_sym_xyz
    
    if args.friedel_symmetrization.status:
        print "#### Generating volume_backg_sub_friedel_sym"
        volume_backg_sub_friedel_sym,weight_friedel_sym = scripts.volume.volume_symmetrize(volume_3d=volume_backg_sub,\
                                                                        volume_mask_3d=vMask, \
                                                                        operator="+++,---",\
                                                                        threshold_keep=(-100,1000),\
                                                                        **args.friedel_symmetrization.params)
        
        volume_backg_sub_friedel_sym[weight_friedel_sym<=0] = -1024
        data_save["volume_backg_sub_friedel_sym"] = volume_backg_sub_friedel_sym
        data_save["weight_friedel_sym"] = weight_friedel_sym
        
        if args.coordinate_conversion.status:
            print "#### Generating volume_backg_sub_friedel_sym_xyz"
            volume_backg_sub_friedel_sym_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_backg_sub_friedel_sym,\
                                                       astar=astar,bstar=bstar,cstar=cstar,\
                                                       volume_mask=(weight_friedel_sym>0).astype(int),\
                                                       threshold_keep=(-100,1000), \
                                                       **args.coordinate_conversion.params) 
            
            data_save["volume_backg_sub_friedel_sym_xyz"] = volume_backg_sub_friedel_sym_xyz

if args.friedel_symmetrization.status:
    print "#### Generating volume_friedel_sym"
    volume_friedel_sym, weight_friedel_sym = scripts.volume.volume_symmetrize(volume_3d=volume,\
                                                                        volume_mask_3d=vMask, \
                                                                        operator="+++,---",\
                                                                        threshold_keep=(0,1000),\
                                                                        **args.friedel_symmetrization.params)
    
    volume_friedel_sym[weight_friedel_sym<=0] = -1024
    data_save["volume_friedel_sym"] = volume_friedel_sym
    data_save["weight_friedel_sym"] = weight_friedel_sym
    
    if args.background_subtraction.status:
        print "#### Generating volume_friedel_sym_backg_sub"
        volume_friedel_sym_backg = scripts.volume.volume_radial_profile(volume_3d=volume_friedel_sym, \
                                                          volume_mask_3d=(weight_friedel_sym>0).astype(int), \
                                                          basis_by_column=Bmat_invA/scripts.utils.length(Bmat_invA[:,1]),\
                                                          threshold_keep=(0,1000), \
                                                          **args.background_subtraction.params)
        
        volume_friedel_sym_backg_sub = volume_friedel_sym - volume_friedel_sym_backg
        volume_friedel_sym_backg_sub[weight_friedel_sym<=0] = -1024
        data_save["volume_friedel_sym_backg_sub"] =  volume_friedel_sym_backg_sub
        
        if args.coordinate_conversion.status:
            print "#### Generating volume_friedel_sym_backg_sub_xyz"
            volume_friedel_sym_backg_sub_xyz, _ = scripts.volume.volume_hkl_reshape(volume=volume_friedel_sym_backg_sub,\
                                                       astar=astar,bstar=bstar,cstar=cstar,\
                                                       volume_mask=(weight_friedel_sym>0).astype(int),\
                                                       threshold_keep=(-100,1000), \
                                                       **args.coordinate_conversion.params)
            
            data_save["volume_friedel_sym_backg_sub_xyz"] = volume_friedel_sym_backg_sub_xyz

            
#### save file
scripts.fsystem.PVmanager.modify(data_save, import_data.get(args.read_dname))
print("#### Map Operate Done")