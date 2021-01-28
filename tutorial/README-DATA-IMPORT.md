dsdata.import
====

### Full parameter command line:
```
dsdata.import \
--fname raw_data.dsdata \
--image_file /DATA/PATH/image_*.cbf \
--backg_file /DATA/PATH/blank_*.cbf \
--gxparms_file /DATA/PATH/GXPARM.XDS \
--dials_report_file /DATA/PATH/dials-report.html \
--detector_mask_file /DATA/PATH/user_mask.npy \
--extra_params /DATA/PATH/extra_json.js \
[variable1=value1] \
[variable2=value2] \
......
```
### Description of each input:  
- ```--fname```:  
All input data files will be recorded in the new file ```fname```. If the ```fname``` already exists, then it won't be rewritten but just updated, so you can run ```dsdata.import``` multiple times to add one input per time, or come to add more files whenever you need.  

- ```--image_file```:  
The experimental data file. The default data format is CBF, but h5py, pickle files are also acceptable, though you need to save your data with the [standard parameter keywords]() for ```dspack```.  

- ```--backg_file```:  
The same as ```--image_file```, for specify the non-crystal background diffraction patterns. You can ignore this input if you don't have background patterns collected.  

- ```--gxparms_file```:  
The XDS indexing result. It's generally saved as GXPARMS.XDS, which contains important geometry and crystal information, such as the detector distance, crystal inverse A matrix, pixel size, etc. If your data are indexed with other softwares such as [CCTBX](https://cci.lbl.gov/cctbx_docs/index.html) or [DIALS](https://dials.github.io/index.html), you should (1) read and diagnose your indexing results, (2) incoporate them to an extra json/yaml file (such as extra_json.js) with the [standard parameter keywords]() for dspack, and then (3) run ```dsdata.import``` to include your own indexing results with ```extra_params /DATA/PATH/extra_json.js```. An example for the user-defined indexing file is [here]().  

- ```--dials_report_file```:  
The dials indexed report file in html format, which is normally saved as dials-report.html. This file contains per-image scale factors for Bragg analysis, and is only used for Bragg intensity scale factor for ```dsimage.scale```. If you don't use the Bragg scale factor, then you don't need to input this.  

- ```--detector_mask_file```:  
The user define the detector mask file in numpy format. The user can provide this file to mask out some detector shadows and artifacts that can't be removed by image processing.  

- ```--extra_params```:  
The backup method for users with other important inputs that are not mentioned before. You can save these parameters to a extra_json.js file, and all these extra parameters will be applied for **every** diffraction pattern. This method are mostly used to input geometry and crystal information, such as the detector thickness, the detector absorption coefficient, etc.  

- ```[variable=value]```:
This is another backup method for users with other files as input. Different from the ```--extra_params``` input, parameters input from ```[variable=value]``` won't be applied for diffraction pattern analysis. It will only be saved to a h5py dataset name for further usage, such as the PDB file. The most common example is shown below.
```
dsdata.import \
--fname raw_data.dsdata\
pdb_file=/PDB/FILE.pdb 
```
