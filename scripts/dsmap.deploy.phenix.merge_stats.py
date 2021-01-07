import os,sys
import numpy as np

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.fsystem
import scripts.volume 
import scripts.statistics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default="./data.dsdata", type=str)
parser.add_argument("-read_dname","--read_dname", help="read dname", default="merge_volume", type=str)
parser.add_argument("-save_dname","--save_dname", help="save dname", default="phenix_merge_stats", type=str)
parser.add_argument("-pdb_dname","--pdb_dname", help="pdb dname", default="pdb_file", type=str)
parser.add_argument("-vmin","--vmin", help="minimum value to keep", default=-100, type=float)
parser.add_argument("-vmax","--vmax", help="maximum value to keep", default=1000, type=float)
args = parser.parse_args()

unique_time_stamp = scripts.utils.get_time()
volume_file = scripts.fsystem.H5manager.reader(args.fname,args.read_dname)
pdb_file = scripts.fsystem.H5manager.reader(args.fname,args.pdb_dname)

workdir = os.path.realpath(os.path.dirname(volume_file))
workdir = os.path.realpath(os.path.join(workdir,"./%s_%s"%(args.save_dname,unique_time_stamp) ))
work_file = os.path.realpath(os.path.join(workdir,"./phenix_data.dsdata"))

if not os.path.isdir(workdir):
    os.makedirs(workdir)

volume_backg_sub = scripts.fsystem.H5manager.reader(volume_file,"volume_backg_sub")
weight = scripts.fsystem.H5manager.reader(volume_file,"weight")
vmask = (weight>=4).astype(int)

# get lattice and symmetry
lat = scripts.statistics.get_pdb_lattice_from_file(pdb_file)
headers = ["    1\n"," -987\n",None]
headers[2] = "%10.4f%10.4f%10.4f%10.3f%10.3f%10.3f %s\n"%(lat[0],lat[1],lat[2],lat[3],lat[4],lat[5],lat[6])
min_value = np.amin(volume_backg_sub * vmask * (volume_backg_sub>args.vmin))
if min_value > 0:
    min_value = 0
elif min_value == 0:
    min_value = -1
else:
    pass 

hkl_file = os.path.realpath(os.path.join(workdir,"./exp_nosym_aniso.hkl"))
scripts.volume.volume_to_phenix(volume=volume_backg_sub,volume_center=None,fsave=hkl_file, vmask=vmask, \
            threshold_keep=(args.vmin, args.vmax), lift=-min_value, headers=headers)

data_save = {}
data_save["exp_volume"] = volume_file
data_save["exp_nosym_aniso_hkl"] = hkl_file
scripts.fsystem.PVmanager.modify(data_save, work_file)

data_save = {}
data_save[args.save_dname] = work_file
scripts.fsystem.PVmanager.modify(data_save, args.fname)
print("#### Phenix WorkDir: %s"%workdir)
print("#### Map Deploy Done")