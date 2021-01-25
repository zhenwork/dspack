import os,sys
import subprocess 
import numpy as np 
from numba import jit 

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.fsystem 
import scripts.volume 

## get results
def get_phenix_overall_cc12(fname):
    with open(fname) as f:
        content = f.readlines()
    for idx,line in enumerate(content):
        if line.startswith("References:"):
            IDX = idx
    cc = float(content[IDX-2].split()[11])
    return cc

def get_phenix_cc12_completeness(path_phenix_dat):
    with open(path_phenix_dat) as f:
        phenix_dat = f.readlines()
    for idx, line in enumerate(phenix_dat):
        if line.startswith(" d_max  d_min"):
            start_idx = idx 
        if line.startswith("References:"):
            stop_idx = idx - 2
    max_res = []
    min_res = []
    complete = []
    cc12 = []
    for line in phenix_dat[(start_idx+1):(stop_idx)]:
        content = line.split()
        if len(content) > 4:
            max_res.append(float(content[1])) # 1.4
            min_res.append(float(content[0])) # 50
            complete.append(float(content[5]))
            cc12.append(float(content[11]))

    # lowres_A_curve,highres_A_curve,phenix_cc12_curve,completeness_curve
    return np.array(min_res), np.array(max_res), np.array(cc12), np.array(complete)
    
    
def get_pdb_lattice_from_file(fpdb):
    with open(fpdb,"r") as f:
        for line in f:
            if line.startswith("CRYST"):
                z = line.split()
                break
    lattice = (float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5]),float(z[6]),"".join(z[7:]).strip())
    return lattice
    
    
def get_llm_cc_from_logs(fllm):
    with open(fllm,"r") as f:
        content = f.readlines()
    cc = - float(content[-4].split()[3])
    gamma = float(content[-1].split()[3])
    sigma = float(content[-1].split()[7])
    return cc,sigma,gamma
    
def get_rbt_cc_from_logs(fllm):
    with open(fllm,"r") as f:
        content = f.readlines()
    cc = -float(content[-4].split()[3])
    gamma = 0
    sigma = float(content[-1].split()[3])
    return cc,sigma,gamma

    
def hkl_file_to_volume(fhkl,volume_size=(121,121,121),volume_center=None):
    nx,ny,nz = volume_size
    volume = np.zeros((nx,ny,nz))-1024
    
    if volume_center is None:
        cx = (nx-1)/2
        cy = (ny-1)/2
        cz = (nz-1)/2
    else:
        cx = volume_center[0]
        cy = volume_center[1]
        cz = volume_center[2]
    
    with open(fhkl,"r") as f:
        content = f.readlines()

    vmax = -1024
    vmin = 1024
    for line in content:
        h,k,l,I = line.split()
        I = float(I)
        vmax = max(vmax, I)
        vmin = min(vmin, I)
    
    if vmax > 0:
        if vmin >= 0:
            multiply_scale = 200. / vmax 
        else:
            multiply_scale = min(200. / vmax, 50./abs(vmin))
    elif vmax == 0:
        if vmin == 0:
            multiply_scale = 1.
        else:
            multiply_scale = 50./abs(vmin)
    else: # vmax<0; vmin<0
        multiply_scale = 50./abs(vmin)
    
    for line in content:
        h,k,l,I = line.split()
        h = int(h)
        k = int(k)
        l = int(l)
        I = float(I)
        if h+cx<0 or h+cx>nx-1:
            continue
        if k+cy<0 or k+cy>ny-1:
            continue
        if l+cz<0 or l+cz>nz-1:
            continue
        volume[h+cx, k+cy, l+cz] = I * multiply_scale
        
    mask = (volume>-1000).astype(int)
    return volume,mask

def run_liquid_like_motion(volume_file,pdb_file,\
                           threshold_keep=(0.01,1000),\
                           max_resolution_A=1.4,\
                           llm_b_factor_mode="three",\
                           workdir=None,\
                           simulation_model="llm"):
    
    volume = scripts.fsystem.H5manager.reader(volume_file,"volume")
    weight = scripts.fsystem.H5manager.reader(volume_file,"weight")
    vmask = (weight>=4).astype(int)
    
    flogs = os.path.realpath(os.path.join(workdir,"./llm_run_llm.sh.%s.out"%simulation_model))
    fsave = os.path.realpath(os.path.join(workdir,"./llm_no_sym_no_sub.hkl"))
    
    scripts.volume.volume_to_txt(volume=volume,volume_center=None,fsave=fsave, vmask=vmask, \
                threshold_keep=threshold_keep, lift=0, headers=None)
    
    lattice = get_pdb_lattice_from_file(pdb_file)[:6]
    unit_cell = ",".join([str(x) for x in lattice])
    command = "sh %s/scripts/run_llm.sh hkl=%s workdir=%s max_res_A=%f pdb_file=%s b_factor=%s unit_cell=%s model=%s >> %s"%(\
                                                    PATH,\
                                                    fsave,\
                                                    workdir,max_resolution_A,\
                                                    pdb_file,llm_b_factor_mode,\
                                                    unit_cell, \
                                                    simulation_model, \
                                                    flogs)
    print "#### RUN LLM COMMAND: ", command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate() 
    process.wait()
    process=None
    
        
    ## get output (cc,gamma,sigma) information
    llm_result = {}
    if llm_b_factor_mode=="three":
        llm_b_factor_modes = ["zero","iso","aniso"]
    else:
        llm_b_factor_modes = [llm_b_factor_mode]
    
    for b_factor_mode in llm_b_factor_modes:
        best_llm_file = os.path.join(workdir,"llm_%s_%s_best.hkl"%(simulation_model,b_factor_mode))
        llm_log_file = os.path.join(workdir,"llm_%s_%s_log.out"%(simulation_model,b_factor_mode))
        
        if simulation_model=="llm":
            llm_correlation_overall,llm_sigma_A,llm_gamma_A = get_llm_cc_from_logs(llm_log_file)
        elif simulation_model=="rbt":
            llm_correlation_overall,llm_sigma_A,llm_gamma_A = get_rbt_cc_from_logs(llm_log_file)
        llm_result["llm_%s_lunus_overall_cc"%b_factor_mode] = llm_correlation_overall
        llm_result["llm_%s_lunus_sigma_A"%b_factor_mode] = llm_sigma_A
        llm_result["llm_%s_lunus_gamma_A"%b_factor_mode] = llm_gamma_A
        llm_result["llm_%s_best_hkl"%b_factor_mode] = best_llm_file
        
    llm_result["llm_run_llm_threshold_keep"] = threshold_keep
    llm_result["llm_run_llm_max_resolution_A"] = max_resolution_A
    llm_result["llm_run_llm_workdir"] = workdir
    return llm_result


def run_llm_correlation(volume_file,pdb_file,\
                        threshold_keep=(-100,1000),\
                        max_resolution_A=1.4,\
                        llm_b_factor_mode="three",\
                        num_resolution_shells=15,\
                        workdir=None):
    
    if llm_b_factor_mode=="three":
        llm_b_factor_modes = ["zero","iso","aniso"]
    else:
        llm_b_factor_modes = [llm_b_factor_mode]
    
    lattice_constant = get_pdb_lattice_from_file(pdb_file)[:6] 
    volume_exp = scripts.fsystem.H5manager.reader(volume_file,"volume_laue_sym_backg_sub")
    weight_exp = scripts.fsystem.H5manager.reader(volume_file,"weight_laue_sym") 
    volume_exp[weight_exp<0.5] = min(threshold_keep[0]-1, -1024) 
    
    llm_result = {}
    for b_factor_mode in llm_b_factor_modes:
        best_llm_file = os.path.realpath(os.path.join(workdir,"llm_%s_best.hkl"%b_factor_mode))
        best_llm_volume, best_llm_mask = hkl_file_to_volume(best_llm_file)
        
        best_llm_volume[best_llm_mask==0] = min(threshold_keep[0]-1, -1024) 
#         print np.amin(best_llm_volume),np.amax(best_llm_volume)
        max_res, min_res, corr, number = scripts.volume.correlation(volume_exp,\
                                                                    best_llm_volume,\
                                                                    lattice_constant_A_deg=lattice_constant,\
                                                                    res_shell_step_Ainv=None,\
                                                                    num_res_shell=num_resolution_shells,\
                                                                    res_cut_off_high_A=max_resolution_A,\
                                                                    res_cut_off_low_A=50,\
                                                                    num_ref_uniform=True,\
                                                                    vmin=threshold_keep[0],\
                                                                    vmax=threshold_keep[1])
        
        ave_res = (1./max_res + 1./min_res)/2. # invA
        ave_res = ave_res[:-1]
        ccllm = corr[:-1]
        
        llm_result["llm_%s_cc_number_per_shell"%b_factor_mode] = number[:-1]
        llm_result["llm_%s_cc_resolution_shell_invA"%b_factor_mode] = ave_res
        llm_result["llm_%s_cc_overall"%b_factor_mode] = corr[-1]
        llm_result["llm_%s_cc_per_shell"%b_factor_mode] = ccllm
    
    llm_result["llm_run_cc_threshold_keep"] = threshold_keep
    llm_result["llm_run_cc_max_resolution_A"] = max_resolution_A
    llm_result["llm_run_cc_workdir"] = workdir
    return llm_result
    
    
def run_phenix_cc12(volume_file,pdb_file,threshold_keep=(-100,1000),max_resolution_A=1.4,\
                           num_resolution_shells=15,workdir=None):
    
    volume_backg_sub = scripts.fsystem.H5manager.reader(volume_file,"volume_backg_sub")
    weight = scripts.fsystem.H5manager.reader(volume_file,"weight")
    vmask = (weight>=4).astype(int)
    
    # get lattice and symmetry
    lat = get_pdb_lattice_from_file(pdb_file)
    headers = ["    1\n"," -987\n",None]
    headers[2] = "%10.4f%10.4f%10.4f%10.3f%10.3f%10.3f %s\n"%(lat[0],lat[1],lat[2],lat[3],lat[4],lat[5],lat[6])
#     print headers[2]
    min_value = np.amin(volume_backg_sub * vmask * (volume_backg_sub>threshold_keep[0]))
    if min_value > 0:
        min_value = 0
    elif min_value == 0:
        min_value = -1
    
    fsave = os.path.realpath(os.path.join(workdir,"./phenix_no_sym_with_sub.hkl"))
    flogs = os.path.realpath(os.path.join(workdir,"./phenix.dat"))
    scripts.volume.volume_to_phenix(volume=volume_backg_sub,volume_center=None,fsave=fsave, vmask=vmask, \
                threshold_keep=threshold_keep, lift=-min_value, headers=headers)
    
    return True
    """
    command = "phenix.merging_statistics file_name=%s high_resolution=%f n_bins=%d >> %s"%(fsave,\
                                                        max_resolution_A,num_resolution_shells, flogs)
    print "#### RUN PHENIX COMMAND:", command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate() 
    process.wait()
    process=None
    
    ## get result
    phenix_result = {}
    ave_res_invA,cc12,completeness = get_phenix_cc12_completeness(flogs)
    cc12_overall = get_phenix_overall_cc12(flogs)
    
    phenix_result["phenix_resolution_shell_invA"] = ave_res_invA
    phenix_result["phenix_cc12_per_shell"] = cc12
    phenix_result["phenix_completeness_per_shell"] = completeness
    phenix_result["phenix_cc12_overall"] = cc12_overall
    phenix_result["phenix_threshold_keep"] = threshold_keep
    phenix_result["phenix_max_resolution_A"] = max_resolution_A
    phenix_result["phenix_workdir"] = workdir
    return phenix_result
    """



def run_cc_intra_inter(data_files=[],pdb_files=[],threshold_keep=(-100,1000),max_resolution_A=1.4,\
                           num_resolution_shells=15):
    
    cc_result = {}
    data_volumes = []
    lattice_constants = []
    num_data = len(data_files)
    for idx,fname in enumerate(data_files):
        data_volumes.append(scripts.fsystem.H5manager.reader(fname,"volume_laue_sym_backg_sub")) 
        lattice_constants.append(get_pdb_lattice_from_file(pdb_files[idx])[:6])
    cc_result["cc_intra_inter_per_shell"] = np.zeros((num_data,num_data,4,num_resolution_shells))
    cc_result["cc_intra_inter"] = np.zeros((num_data,num_data,2))
    for i in range(num_data):
        for j in range(num_data):
            if i>j:
                continue
            lattice_constant = np.array(lattice_constants[i])/2.+np.array(lattice_constants[j])/2.
            max_res, min_res, corr, number = scripts.volume.correlation(data_volumes[i], \
                                                                    data_volumes[j], \
                                                                    lattice_constant_A_deg=lattice_constant, \
                                                                    res_shell_step_Ainv=None, \
                                                                    num_res_shell=num_resolution_shells, \
                                                                    res_cut_off_high_A=max_resolution_A, \
                                                                    res_cut_off_low_A=50, \
                                                                    num_ref_uniform=True, \
                                                                    vmin=threshold_keep[0], \
                                                                    vmax=threshold_keep[1])
            
            cc_result["cc_intra_inter_per_shell"][i,j,0] = max_res[:-1]
            cc_result["cc_intra_inter_per_shell"][i,j,1] = min_res[:-1]
            cc_result["cc_intra_inter_per_shell"][i,j,2] = corr[:-1]
            cc_result["cc_intra_inter_per_shell"][i,j,3] = number[:-1]
            cc_result["cc_intra_inter"][i,j,0] = corr[-1]
            cc_result["cc_intra_inter"][i,j,1] = number[-1]
            
            
    cc_result["cc_intra_inter_threshold_keep"] = threshold_keep
    cc_result["cc_intra_inter_max_resolution_A"] = max_resolution_A 
    return cc_result

def run_visualization(volume_file,threshold_keep=(-100,1000),radius_max_A=1.4,radius_min_A=50):
    results = {} 
    data = scripts.fsystem.H5manager.reader(volume_file,"volume_laue_sym_backg_sub")
    Bmat_invA = scripts.fsystem.H5manager.reader(volume_file,"Bmat_invA")
    oversample = scripts.fsystem.H5manager.reader(volume_file,"oversample") 
    voxel_size = Bmat_invA[1,1]
    radius_max_px = 1./radius_max_A/voxel_size * oversample
    if radius_min_A is not None:
        radius_min_px = 1./radius_min_A/voxel_size * oversample
    else:
        radius_min_px = None
        
    nx,ny,nz = data.shape
    cx,cy,cz = (nx-1)/2,(ny-1)/2,(nz-1)/2
    
    detvalue, detcount, detpolar, resolution, cenX, cenY, proj = \
                   scripts.volume.pViewer(_volume=data, vector=(1,2,1), center=(cx,cy,cz), directions=None, \
                                               voxelsize=1,vmin=threshold_keep[0], vmax=threshold_keep[1], \
                                               rmax=radius_max_px, rmin=radius_min_px, \
                                               depth=3, stretch=1.0, standardCut=True)
    
    results["visualization_max_resolution_A"] = radius_max_A
    results["visualization_threshold_keep"] = threshold_keep
    
    #  
    results["visualization_detvalue"] = detvalue 
    results["visualization_detcount"] = detcount 
    results["visualization_detpolar"] = detpolar 
    results["visualization_resolution"] = resolution 
    results["visualization_cenX"] = cenX 
    results["visualization_cenY"] = cenY 
    results["visualization_proj"] = proj 
    return results 

import h5py
def calculate_ccfriedel_from_map(map_file=None,pdb_file=None,rmin_A=50.,rmax_A=1.4,vmin=-100,vmax=1000):
    path_volume = map_file
    pdb_file = pdb_file
    with h5py.File(path_volume) as f:
        volume_backg_sub = f["volume_backg_sub"].value 
        volume_backg_sub_friedel_sym = f["volume_backg_sub_friedel_sym"].value 
        weight = f["weight"].value 

    vmask = (weight>=4).astype(int)
    volume_backg_sub[vmask==0] = -1024
    volume_backg_sub_friedel_sym[vmask==0] = -1024
    lattice_constant = get_pdb_lattice_from_file(pdb_file)[:6]
    max_res, min_res, corr, number = scripts.volume.correlation(volume_backg_sub, \
                                                                volume_backg_sub_friedel_sym, \
                                                                lattice_constant_A_deg=lattice_constant, \
                                                                res_shell_step_Ainv=None, \
                                                                num_res_shell=15, \
                                                                res_cut_off_high_A=rmax_A, \
                                                                res_cut_off_low_A=rmin_A, \
                                                                num_ref_uniform=True, \
                                                                vmin=vmin, \
                                                                vmax=vmax)
    return corr[-1]


def calculate_cclaue_from_map(map_file=None,pdb_file=None,rmin_A=50.,rmax_A=1.4,vmin=-100,vmax=1000):
    path_volume = map_file
    pdb_file = pdb_file
    with h5py.File(path_volume) as f:
        volume_backg_sub = f["volume_backg_sub"].value 
        volume_backg_sub_laue_sym = f["volume_backg_sub_laue_sym"].value 
        weight = f["weight"].value 

    vmask = (weight>=4).astype(int)
    volume_backg_sub[vmask==0] = -1024
    volume_backg_sub_laue_sym[vmask==0] = -1024
    lattice_constant = get_pdb_lattice_from_file(pdb_file)[:6]
    max_res, min_res, corr, number = scripts.volume.correlation(volume_backg_sub, \
                                                                volume_backg_sub_laue_sym, \
                                                                lattice_constant_A_deg=lattice_constant, \
                                                                res_shell_step_Ainv=None, \
                                                                num_res_shell=15, \
                                                                res_cut_off_high_A=rmax_A, \
                                                                res_cut_off_low_A=rmin_A, \
                                                                num_ref_uniform=True, \
                                                                vmin=vmin, \
                                                                vmax=vmax)
    return corr[-1]