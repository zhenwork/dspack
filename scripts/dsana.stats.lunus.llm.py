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
parser.add_argument("-lunus_data","--lunus_data", help="lunus_llm", default="lunus_llm", type=str)
parser.add_argument("--launch_lunus_model",help="params",default=None,nargs="*")
args = parser.parse_args()

unique_time_stamp = scripts.utils.get_time()
lunus_data_file = scripts.fsystem.H5manager.reader(args.fname,args.lunus_data)
hkl_file = scripts.fsystem.H5manager.reader(lunus_data_file,"exp_nosym_nosub_hkl")
pdb_file = scripts.fsystem.H5manager.reader(lunus_data_file,"pdb_file")
workdir = os.path.realpath(os.path.dirname(lunus_data_file))

##############################################################
args.launch_lunus_model = scripts.utils.get_process_params(args.launch_lunus_model)
max_resolution_A = args.launch_lunus_model.params.get("max_resolution_A") or 1.4
b_factor_mode = args.launch_lunus_model.params.get("b_factor_mode") or "three"
model = args.launch_lunus_model.params.get("model") or "llm"
symop = args.launch_lunus_model.params.get("symop") or -1
##############################################################

log_file = os.path.realpath(os.path.join(workdir,"./dsana.stats.lunus.llm_%s_%s_%s.out"%(model,b_factor_mode,unique_time_stamp)))
lattice = scripts.statistics.get_pdb_lattice_from_file(pdb_file)[:6]
unit_cell = ",".join([str(x) for x in lattice])
command = "sh %s/app/dsana.stats.lunus.llm.helper hkl=%s workdir=%s max_res_A=%f pdb_file=%s b_factor=%s unit_cell=%s model=%s symop=%d >> %s"%(\
                                                PATH,\
                                                hkl_file,\
                                                workdir,\
                                                max_resolution_A,\
                                                pdb_file,\
                                                b_factor_mode,\
                                                unit_cell, \
                                                model, \
                                                symop, \
                                                log_file)
print "#### RUN LLM COMMAND: ", command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, err = process.communicate() 
process.wait()
process=None

## lunus_llm_phenix_// lunus_rbt_three_tag.out lunus_rbt_zero_sym_aniso_tag.hkl
llm_result = {}
if b_factor_mode=="three":
    b_factor_modes = ["zero","iso","aniso"]
else:
    b_factor_modes = [b_factor_mode]

for b_factor_mode in b_factor_modes:
    lunus_sym_aniso_hkl = os.path.join(workdir,"lunus_%s_%s_sym_aniso.hkl"%(model,b_factor_mode))
    lunus_nosym_nosub_lat = os.path.join(workdir,"lunus_%s_%s_nosym_nosub.lat"%(model,b_factor_mode))
    lunus_sym_nosub_lat = os.path.join(workdir,"lunus_%s_%s_sym_nosub.lat"%(model,b_factor_mode))
    lunus_sym_aniso_lat = os.path.join(workdir,"lunus_%s_%s_sym_aniso.lat"%(model,b_factor_mode))
    lunus_stats_file = os.path.join(workdir,"lunus_%s_%s_stats.out"%(model,b_factor_mode))

    if model=="llm":
        lunus_cc_overall,lunus_sigma_A,lunus_gamma_A = scripts.statistics.get_llm_cc_from_logs(lunus_stats_file)
    elif model=="rbt":
        lunus_cc_overall,lunus_sigma_A,lunus_gamma_A = scripts.statistics.get_rbt_cc_from_logs(lunus_stats_file)
    
    llm_result["lunus_%s_%s_cc"%(model,b_factor_mode)] = lunus_cc_overall
    llm_result["lunus_%s_%s_sigma_A"%(model,b_factor_mode)] = lunus_sigma_A
    llm_result["lunus_%s_%s_gamma_A"%(model,b_factor_mode)] = lunus_gamma_A
    llm_result["lunus_%s_%s_sym_aniso_hkl"%(model,b_factor_mode)] = lunus_sym_aniso_hkl
    llm_result["lunus_%s_%s_nosym_nosub_lat"%(model,b_factor_mode)] = lunus_nosym_nosub_lat
    llm_result["lunus_%s_%s_sym_nosub_lat"%(model,b_factor_mode)] = lunus_sym_nosub_lat
    llm_result["lunus_%s_%s_sym_aniso_lat"%(model,b_factor_mode)] = lunus_sym_aniso_lat
    llm_result["lunus_%s_%s_stats_file"%(model,b_factor_mode)] = lunus_stats_file

###############
scripts.fsystem.PVmanager.modify(llm_result, lunus_data_file)
print("#### Lunus Results: %s"%workdir)
print("#### Lunus Analysis Done")