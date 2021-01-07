
import os,sys
import numpy as np

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.fsystem
import scripts.statistics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files",default=None,nargs="*")
parser.add_argument("-fsave","--fsave", help="save file name", default="dsana.stats.cc.map.out", type=str)
parser.add_argument("-read_dname","--read_dname", help="volume dname", default="merge_volume", type=str)
parser.add_argument("-pdb_dname","--pdb_dname", help="pdb dname", default="pdb_file", type=str)
parser.add_argument("-nshells","--nshells", help="num shells", default=15, type=int)
parser.add_argument("-vmin","--vmin", help="min value", default=-100, type=float)
parser.add_argument("-vmax","--vmax", help="max value", default=1000, type=float)
parser.add_argument("-rmax_A","--rmax_A", help="max res", default=1.4, type=float)
parser.add_argument("-rmin_A","--rmin_A", help="min res", default=50, type=float)
args = parser.parse_args()

if not args.fname:
    sys.exit(1)
if len(args.fname)<2:
    sys.exit(1)

data_files = []
pdb_files = []
for fname in args.fname:
    volume_file = scripts.fsystem.H5manager.reader(fname,args.read_dname)
    pdb_file = scripts.fsystem.H5manager.reader(fname,args.pdb_dname)
    data_files.append(volume_file)
    pdb_files.append(pdb_file)

cc_result = {}
data_volumes = []
lattice_constants = []
num_data = len(data_files)
for idx,fname in enumerate(data_files):
    data_volumes.append(scripts.fsystem.H5manager.reader(fname,"volume_laue_sym_backg_sub")) 
    lattice_constants.append(scripts.statistics.get_pdb_lattice_from_file(pdb_files[idx])[:6])

cc_result["rmax_A_curve"] = np.zeros((num_data,num_data,args.nshells))
cc_result["rmin_A_curve"] = np.zeros((num_data,num_data,args.nshells))
cc_result["cc_curve"] = np.zeros((num_data,num_data,args.nshells))
cc_result["cc"] = np.zeros((num_data,num_data))

for i in range(num_data):
    for j in range(num_data):
        if i>j:
            continue
        lattice_constant = np.array(lattice_constants[i])/2.+np.array(lattice_constants[j])/2.
        max_res, min_res, corr, number = scripts.volume.correlation(data_volumes[i], \
                                                                    data_volumes[j], \
                                                                    lattice_constant_A_deg=lattice_constant, \
                                                                    res_shell_step_Ainv=None, \
                                                                    num_res_shell=args.nshells, \
                                                                    res_cut_off_high_A=args.rmax_A, \
                                                                    res_cut_off_low_A=args.rmin_A, \
                                                                    num_ref_uniform=True, \
                                                                    vmin=args.vmin, \
                                                                    vmax=args.vmax)
        
        cc_result["rmax_A_curve"][i,j] = max_res[:-1]
        cc_result["rmin_A_curve"][i,j] = min_res[:-1]
        cc_result["cc_curve"][i,j] = corr[:-1]
        cc_result["cc"][i,j] = corr[-1]

cc_result["map_file"] = np.array(data_files)
cc_result["pdb_file"] = np.array(pdb_files)
cc_result["vmin"] = args.vmin
cc_result["vmax"] = args.vmax
cc_result["rmin_A"] = args.rmin_A
cc_result["rmax_A"] = args.rmax_A

tmpdata = np.around(cc_result["cc"],2)
print "#### CC stats"
print tmpdata
###############
scripts.fsystem.PVmanager.modify(cc_result, args.fsave)
print("#### CC Results: %s"%args.fsave)
print("#### CC Analysis Done")