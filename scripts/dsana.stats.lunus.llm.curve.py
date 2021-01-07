import os,sys
import numpy as np
import subprocess 

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.fsystem
import scripts.statistics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default="./data.dsdata", type=str)
parser.add_argument("-read_dname","--read_dname", help="lunus_llm", default="lunus_llm", type=str)
parser.add_argument("-b_factor_mode","--b_factor_mode", help="b factor mode", default="three", type=str)
parser.add_argument("-nshells","--nshells", help="num shells", default=15, type=int)
parser.add_argument("-vmin","--vmin", help="min value", default=-100, type=float)
parser.add_argument("-vmax","--vmax", help="max value", default=1000, type=float)
parser.add_argument("-rmax_A","--rmax_A", help="max res", default=1.4, type=float)
parser.add_argument("-rmin_A","--rmin_A", help="min res", default=50, type=float)
args = parser.parse_args()

unique_time_stamp = scripts.utils.get_time()
lunus_data_file = scripts.fsystem.H5manager.reader(args.fname,args.read_dname)
volume_file = scripts.fsystem.H5manager.reader(lunus_data_file,"exp_volume")
pdb_file = scripts.fsystem.H5manager.reader(lunus_data_file,"pdb_file")
workdir = os.path.realpath(os.path.dirname(lunus_data_file))

lattice_constant = scripts.statistics.get_pdb_lattice_from_file(pdb_file)[:6] 
volume_exp = scripts.fsystem.H5manager.reader(volume_file,"volume_laue_sym_backg_sub")
weight_exp = scripts.fsystem.H5manager.reader(volume_file,"weight_laue_sym") 
volume_exp[weight_exp<0.5] = min(args.vmin-1, -1024) 

if args.b_factor_mode == "three":
    b_factor_modes = ["zero","iso","aniso"]
else:
    b_factor_modes = [args.b_factor_mode]

llm_result = {}
for b_factor_mode in b_factor_modes:
    print "#### Running b factor mode: %s"%b_factor_mode
    best_llm_file = os.path.join(workdir,"lunus_llm_%s_sym_aniso.hkl"%(b_factor_mode))
    best_llm_volume, best_llm_mask = scripts.statistics.hkl_file_to_volume(best_llm_file)
    print "#### llm sym aniso map: min/max=", np.amin(best_llm_volume), np.amax(best_llm_volume)
    
    best_llm_volume[best_llm_mask==0] = min(args.vmin-1, -1024)
    max_res, min_res, corr, number = scripts.volume.correlation(volume_exp,\
                                                                best_llm_volume,\
                                                                lattice_constant_A_deg=lattice_constant,\
                                                                res_shell_step_Ainv=None,\
                                                                num_res_shell=args.nshells,\
                                                                res_cut_off_high_A=args.rmax_A,\
                                                                res_cut_off_low_A=args.rmin_A,\
                                                                num_ref_uniform=True,\
                                                                vmin=args.vmin,\
                                                                vmax=args.vmax)

    ave_res = (1./max_res + 1./min_res)/2. # invA
    ave_res = ave_res[:-1]
    ccllm = corr[:-1]
    
    llm_result["lunus_llm_%s_ccllm_curve"%(b_factor_mode)]  = ccllm
    llm_result["lunus_llm_%s_rmax_A_curve"%(b_factor_mode)] = max_res[:-1]
    llm_result["lunus_llm_%s_rmin_A_curve"%(b_factor_mode)] = min_res[:-1]
    
###############
scripts.fsystem.PVmanager.modify(llm_result, lunus_data_file)
print("#### Lunus Results: %s"%workdir)
print("#### Lunus Analysis Done")