
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
parser.add_argument("-read_dname","--read_dname", help="phenix dname", default="phenix_merge_stats", type=str)
parser.add_argument("-rmax_A","--rmax_A", help="max resolution", default=1.4, type=float)
parser.add_argument("-nshells","--nshells", help="num shells", default=15, type=int)
args = parser.parse_args()

unique_time_stamp = scripts.utils.get_time()
phenix_data_file = scripts.fsystem.H5manager.reader(args.fname,args.read_dname)
hkl_file = scripts.fsystem.H5manager.reader(phenix_data_file,"exp_nosym_aniso_hkl")
workdir = os.path.realpath(os.path.dirname(phenix_data_file))

phenix_stats_file = os.path.realpath(os.path.join(workdir,"phenix.merging_statistics.out"))
command = "rm -f %s && phenix.merging_statistics file_name=%s high_resolution=%f n_bins=%d >> %s"%(phenix_stats_file, hkl_file,\
                                                        args.rmax_A,args.nshells,phenix_stats_file)
print "#### RUN LLM COMMAND: ", command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, err = process.communicate() 
process.wait()
process=None

## 
phenix_result = {}
lowres_A_curve,highres_A_curve,phenix_cc12_curve,completeness_curve = scripts.statistics.get_phenix_cc12_completeness(phenix_stats_file)
phenix_cc12_overall = scripts.statistics.get_phenix_overall_cc12(phenix_stats_file)

phenix_result["phenix_rmax_A"] = args.rmax_A
phenix_result["phenix_cc12"] = phenix_cc12_overall
phenix_result["phenix_cc12_curve"] = phenix_cc12_curve
phenix_result["phenix_rmin_A_curve"] = lowres_A_curve
phenix_result["phenix_rmax_A_curve"] = highres_A_curve
phenix_result["phenix_completeness_curve"] = completeness_curve

###############
scripts.fsystem.PVmanager.modify(phenix_result, phenix_data_file)
print("#### Phenix Results: %s"%workdir)
print("#### Phenix Analysis Done")