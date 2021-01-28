dsdata.import
====

### Full parameter command line:
```
dsdata.import \
--fname ./data.dsdata \
--image_file /PATH/TO/image_*.cbf \
--backg_file /PATH/TO/blank_*.cbf \
--gxparms_file /PATH/TO/GXPARM.XDS \
--dials_report_file /PATH/TO/dials-report.html \
--detector_mask_file /PATH/TO/user_mask.npy \
--extra_params /PATH/TO/extra_json.js \
[variable1=value1] \
[variable2=value2] \
......
```
### Description of each input:  
- ```--fname```:  
All input data files will be recorded to file ```--fname```. If the ```--fname``` already exists, then it won't be rewritten but just updated, so you can run ```dsdata.import``` multiple times to add necessary data files whenever you need. The default is ```./data.dsdata```.  

- ```--image_file```:  
The experimental crystal diffraction data files. The default data format is CBF, but h5py, pickle files are also acceptable, though you need to save your data with the [standard parameter keywords](https://github.com/zhenwork/dspack/blob/main/tutorial/README-STANDARD-KEYWORDS.md) for ```dspack```.  

- ```--backg_file```:  
The same as ```--image_file```, to specify the non-crystal background diffraction patterns. You can ignore this if you don't have background patterns collected.  

- ```--gxparms_file```:  
The XDS indexing result. It's generally saved as GXPARMS.XDS, which contains important geometry and crystal information, such as the detector distance, crystal inverse A matrix, detector center, etc. If your data are indexed with other software such as [CCTBX](https://cci.lbl.gov/cctbx_docs/index.html) or [DIALS](https://dials.github.io/index.html), you should (1) read and diagnose your indexing results, (2) incorporate and save them to an extra json/yaml/pickle file (such as extra_json.js) with the [standard parameter keywords](https://github.com/zhenwork/dspack/blob/main/tutorial/README-STANDARD-KEYWORDS.md), and then (3) run ```dsdata.import``` to include your own indexing results with ```--extra_params /PATH/TO/extra_json.js```. An example for the user-defined extra parameter file is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/example/extra_json.js).  

- ```--dials_report_file```:  
The dials indexing report file in html format, which is normally saved as dials-report.html. This file contains per-image scale factors for Bragg peaks, and is only used to calculate the Bragg intensity scale factor in [```dsimage.scale```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-SCALE.md) step. If you don't use the Bragg scale factor, then you don't need to this input.  

- ```--detector_mask_file```:  
The user-defined detector mask file in numpy format. The user can provide this file to mask out some detector shadows and artifacts that can't be recognized and removed by the standard image processing methods.  

- ```--extra_params```:  
The backup method for users with other important inputs that are not mentioned before. You can save your additional parameters to an extra_json.js file, and these extra parameters will be loaded and applied for **every** single diffraction pattern. This method are mostly used to input geometry and crystal information, such as the detector thickness, the detector absorption coefficient, etc.  

- ```[variable=value]```:
This is another backup method for users with more files as input. Different from the ```--extra_params``` input, parameters in ```[variable=value]``` won't be applied for every diffraction pattern. It will only be recorded to the ```--fname``` file for further usage, such as multiple PDB files. An example is shown below.
```
  dsdata.import \
  --fname raw_data.dsdata \
  pdb_phenix=/PATH/TO/phenix.pdb \
  pdb_refmac5=/PATH/TO/refmac5.pdb
```
