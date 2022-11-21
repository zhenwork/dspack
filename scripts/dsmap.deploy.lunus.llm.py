import os,sys
import numpy as np

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.fsystem
import scripts.volume 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default="./data.dsdata", type=str)
parser.add_argument("-read_dname","--read_dname", help="merge dname", default="merge_volume", type=str)
parser.add_argument("-pdb_dname","--pdb_dname", help="pdb dname", default="pdb_file", type=str)
parser.add_argument("-lunus_data","--lunus_data", help="create dset", default="lunus_llm", type=str)
args = parser.parse_args()

unique_time_stamp = scripts.utils.get_time()
volume_file = scripts.fsystem.H5manager.reader(args.fname,args.read_dname)
pdb_file = scripts.fsystem.H5manager.reader(args.fname,args.pdb_dname)

workdir = os.path.realpath(os.path.dirname(volume_file))
workdir = os.path.realpath(os.path.join(workdir,"./%s_%s"%(args.lunus_data,unique_time_stamp) ))
work_file = os.path.realpath(os.path.join(workdir,"./lunus_data.dsdata"))

if not os.path.isdir(workdir):
    os.makedirs(workdir)

volume = scripts.fsystem.H5manager.reader(volume_file,"volume")
weight = scripts.fsystem.H5manager.reader(volume_file,"weight")
vmask = (weight>=4).astype(int)

hkl_file = os.path.realpath(os.path.join(workdir,"./exp_nosym_nosub.hkl"))

scripts.volume.volume_to_txt(volume=volume, volume_center=None, fsave=hkl_file, vmask=vmask, \
            threshold_keep=(0.01,1000), lift=0, headers=None)


data_save = {}
data_save["pdb_file"] = pdb_file
data_save["exp_volume"] = volume_file
data_save["exp_nosym_nosub_hkl"] = hkl_file
scripts.fsystem.PVmanager.modify(data_save, work_file)

data_save = {}
data_save[args.lunus_data] = work_file
scripts.fsystem.PVmanager.modify(data_save, args.fname)
print("#### Lunus WorkDir: %s"%workdir)
print("#### Map Deploy Done")