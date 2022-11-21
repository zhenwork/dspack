Get started with dspack
=====================
- [Purpose](#purpose)
- [Requirements](#requirements)
	- [Conda packages](#conda-packages)
	- [Software](#software)
- [Setup](#setup)
- [Workflow](#workflow)
	- [Data import](#data-import)
	- [Image clean](#image-clean)
	- [Image scale](#image-scale)
	- [Radial profile variance removal](#radial-profile-variance-removal)
	- [Merge to volume](#merge-to-volume)
	- [Volume operations](#volume-operations)
	- [Statistics](#statistics)
		- [Lunus LLM](#lunus-llm)
		- [PHENIX CChalf](#phenix-cchalf)
		- [CC](#cc)
    
# Purpose
This package ```dspack``` is developed to analyze the diffuse scattering data. The main goal is to build a simple and flexible pipeline to extract the 3D diffuse map from a set of 2D diffraction patterns. This package will read diffraction patterns and other supporting files, perform per-image preprocesses, map and merge 2D patterns into a 3D diffraction volume, process to the diffraction volume to extract symmetrized anisotropic diffuse data, evaluate the diffuse data quality, and model the diffuse map with the liquid-like motions (LLM) model. This package supports options to turn on or turn off each substep, and also accepts different input variables from the users, enabling us to compare various data processing choices. Currently this package was only tested for the ICH dataset.

# Requirements
### Conda packages
The required conda packages are as shown below. They are useful in data reading/saving, image preprocessing, and volume operations, etc. Among them, ```cbf``` is used to read the experimental data, and ```numba``` is used to speed up the process, where the analysis can be slow for large images and large datasets. MPI is also supported for several time-consuming steps.
<details><summary>Conda Packages</summary>
<p>
   
```
	re
	cbf
	json
	yaml
	math
	h5py
	copy
	numpy
	scipy
	numba
	shlex
	os,sys
	mpi4py
	sklearn
	subprocess
	matplotlib
	time,datetime
```
</p>
</details>

### Software
First of all, no external software are required if the final goal is only to generate the 3D diffuse map. However, to evaluate the diffuse data, [PHENIX](https://www.phenix-online.org) is required for the calculation of CChalf, and [Lunus](https://github.com/mewall/lunus) is required to run the liquid-like motions (LLM) model. For psana users, Lunus can be sourced directly from [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-LUNUS-PSANA.md).
- [PHENIX](https://www.phenix-online.org)
- [Lunus](https://github.com/mewall/lunus)

# Setup
To set up ```dspack```, simply git clone the package and source the setup.sh, as shown below. The ```DSPACK_HOME``` will be added to your path. Be sure to also activate the required conda environment and software.
```
git clone https://github.com/zhenwork/dspack.git
cd dspack
chmod +x setup.sh
source setup.sh
```

# Workflow
### Data import
This step uses the [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) method, which is to import the experimental data, geometry files, indexing results, and other extra files. This step won't read out the real data, but just record required files for further process. Some files are only required for diffuse data quality analysis and not necessary for image preprocessing, so you can then run ```dsdata.import``` again to add these files later. The command line is shown below, and the detailed explanation of the full-parameter command is [```here```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md).
```
dsdata.import \
--fname raw_data.dsdata \
--image_file /PATH/TO/image_*.cbf \
--backg_file /PATH/TO/blank_*.cbf \
--gxparms_file /PATH/TO/GXPARM.XDS \
--detector_mask_file /PATH/TO/user_mask.npy \
--extra_params /PATH/TO/extra_json.js
```

### Image clean
This step uses the [```dsimage.clean```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-CLEAN.md) method, which includes **five** image preprocessing substeps: (1) the user-defined detector masking, (2) a deeper bad pixel removal, (3) non-crystal background image subtraction, (4) pixel intensity and position corrections, and (5) Bragg peak cleaning. The 6th image preprocessing substep (scaling and radial profile variance removal) is designed separately to save time for testing multiple scale factors. The command line for [```dsimage.clean```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-CLEAN.md) is shown below. Note that this process is quite slow, be sure to use MPI to speed it up by, for example, using ```mpirun -n 10 dsimage.clean```
```
dsimage.clean \
--fname raw_data.dsdata \
--fsave cleaned_data.dsdata \
--apply_detector_mask \
--remove_bad_pixels \
--subtract_background \
--parallax_correction \
--polarization_correction \
--solid_angle_correction \
--detector_absorption_correction \
--remove_bragg_peaks
```
To turn off a substep in the command line, you can simply remove the name of that substep, for example, remove ```--polarization_correction``` to turn off the polarization correction step, and remove ```--solid_angle_correction``` to turn off the solid angle correction. The corresponding command lines are shown below. You can also turn off multiple substeps if you want.
<details><summary>Turn off the polarization correction</summary>
<p>
   
```
dsimage.clean \
--fname raw_data.dsdata \
--fsave cleaned_data.dsdata \
--apply_detector_mask \
--remove_bad_pixels \
--subtract_background \
--parallax_correction \
--solid_angle_correction \
--detector_absorption_correction \
--remove_bragg_peaks
```
</p>
</details>
<details><summary>Turn off the solid angle correction</summary>
<p>
   
```
dsimage.clean \
--fname raw_data.dsdata \
--fsave cleaned_data.dsdata \
--apply_detector_mask \
--remove_bad_pixels \
--subtract_background \
--parallax_correction \
--polarization_correction \
--detector_absorption_correction \
--remove_bragg_peaks
```
</p>
</details>

These simplified command lines will use default image processing parameters, whose functionality is totally the same as the full-parameter command line as shown below. Users can change the input parameters for each substep. From the full-parameter command line, it's quite easy to guess what each substep is doing. The detailed description of each substep is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-CLEAN.md).
<details><summary>Full parameter command line</summary>
<p>
   
```
dsimage.clean \
--fname raw_data.dsdata \
--fsave cleaned_data.dsdata \
--apply_detector_mask radius_rmin_px=40 radius_rmax_px=None value_vmin=0 value_vmax=10000 \
--remove_bad_pixels algorithm=2 sigma=5 window_size_px=11 \
--subtract_background \
--parallax_correction \
--polarization_correction \
--solid_angle_correction \
--detector_absorption_correction \
--remove_bragg_peaks replace_by_median=True window_size_px=11
```
</p>
</details>

### Image scale
Here we provide four different per-image scale factors, which are radial profile, water ring, overall, and Bragg scale factors. The standard pipeline uses the radial profile scale factor, whose command line is shown below.  
```
dsimage.scale \
--fname cleaned_data.dsdata \
--fsave cleaned_data_scaled.dsdata \
--scale_by_radial_profile
```
To use other scale factors, simply indicate the name of that scale factor, as shown below.
<details><summary>Water ring scale factor</summary>
<p>
   
```
dsimage.scale \
--fname cleaned_data.dsdata \
--fsave cleaned_data_scaled.dsdata \
--scale_by_water_ring_intensity
```
</p>
</details>
<details><summary>Overall scale factor</summary>
<p>
   
```
dsimage.scale \
--fname cleaned_data.dsdata \
--fsave cleaned_data_scaled.dsdata \
--scale_by_overall_intensity
```
</p>
</details>
<details><summary>Bragg intensity scale factor</summary>
<p>
   
```
dsimage.scale \
--fname cleaned_data.dsdata \
--fsave cleaned_data_scaled.dsdata \
--scale_by_bragg_intensity
```
</p>
</details>

These are simplified command lines with default parameters, the full-parameter command line and detailed explanations for input parameters are shown [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-SCALE.md).  


### Radial profile variance removal
This step removes the variance of the radial intensity profiles of a dataset. The radial profile variance is calculated using three largest PCA components, first introduced by [Peck et al.](https://doi.org/10.1107/S2052252518001124) The command line is shown below, and the detailed explanation for each parameter is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-PCA.md).
```
dsimage.pca \
--fname clean_data_scaled.dsdata \
--fsave clean_data_scaled_with_pca.dsdata \
--radial_pca_subtraction
```
Up to now, all **six** image preprocessing steps have been applied to every image, we will then merge them to the 3D reciprocal space.

### Merge to volume
This step merges all preprocessed diffraction patterns into a 3D diffraction volume. The command line is shown below, and the detailed explanation for each parameter is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-MERGE.md).  
```
dsmap.merge \
--fname clean_data_scaled_with_pca.dsdata \
--fsave clean_data_scaled_with_pca_map.dsdata \
--merge_to_volume
```

### Volume operations
This step performs Laue-/Friedel- symmetrization and isotropic component subtraction to the raw diffraction volume from the previous step [```dsmap.merge```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-MERGE.md). The command line is shown below, and detailed explanation is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-OPERATE.md).
```
dsmap.operate \
--fname clean_data_scaled_with_pca_map.dsdata \
--laue_symmetrization \
--friedel_symmetrization \
--background_subtraction
```

### Statistics

##### Lunus LLM  

To run the LLM model, we need to provide the PDB file, export the 3D diffuse map to the [Lunus](https://github.com/mewall/lunus) accepted format. The command lines are shown below, and the detailed explanations for data export and analysis are [```dsmap.deploy.lunus.llm```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md) and [```dsana.stats.lunus.llm```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DSANA.md).
<details><summary>Import the PDB file</summary>
<p>
   
```
dsdata.import \
--fname clean_data_scaled_with_pca_map.dsdata \
pdb_refmac5=/PATH/TO/protein.pdb 
```
</p>
</details>
<details><summary>Export diffuse map for Lunus</summary>
<p>
   
```
dsmap.deploy.lunus.llm \
--fname clean_data_scaled_with_pca_map.dsdata \
--pdb_dname pdb_refmac5 \
--lunus_data lunus_llm_refmac5
```
</p>
</details>
<details><summary>Run LLM model</summary>
<p>
   
```
dsana.stats.lunus.llm \
--fname clean_data_scaled_with_pca_map.dsdata \
--lunus_data lunus_llm_refmac5 \
--launch_lunus_model b_factor_mode=aniso
```
</p>
</details>

##### PHENIX CChalf  

Similarly, to run [PHENIX](https://www.phenix-online.org) for CChalf, you need to import the PDB file, export the unsymmetrized anisotropic map to the [PHENIX](https://www.phenix-online.org) accepted format. The command lines are shown below, and the detailed explanations for data export and analysis are [```dsmap.deploy.phenix.merge_stats```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md) and [```dsana.stats.phenix.merge_stats```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DSANA.md).
<details><summary>Import the PDB file</summary>
<p>
   
```
dsdata.import \
--fname clean_data_scaled_with_pca_map.dsdata \
pdb_refmac5=/PATH/TO/PDB.pdb 
```
</p>
</details>
<details><summary>Export diffuse map for PHENIX</summary>
<p>
   
```
dsmap.deploy.phenix.merge_stats \
--fname clean_data_scaled_with_pca_map.dsdata \
--pdb_dname pdb_refmac5
```
</p>
</details>
<details><summary>Run PHENIX</summary>
<p>
   
```
dsana.stats.phenix.merge_stats \
--fname clean_data_scaled_with_pca_map.dsdata
```
</p>
</details>

##### CC  

To calculate the Pearson correlation coefficient (CC) between diffuse maps of several datasets, you can simply use any python scripts with general CC functions, however, you can only use intensities in common voxels of diffuse map pairs. The command line for this calculation is shown below, and detailed parameters are discussed [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-STATS.md).
```
dsana.stats.cc.map \
--fname clean_data_scaled_with_pca_map.dsdata \
--read_dname merge_volume \
--pdb_dname pdb_refmac5
```
