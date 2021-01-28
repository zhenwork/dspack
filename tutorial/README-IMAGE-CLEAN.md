dsdata.image
=====

### Full parameter command line:
```
dsimage.clean \
--fname raw_data.dsdata \
--fsave cleaned_data.dsdata \
--read_dname image_file \
--apply_detector_mask radius_rmin_px=40 radius_rmax_px=None value_vmin=0 value_vmax=10000 \
--remove_bad_pixels algorithm=2 sigma=5 window_size_px=11 \
--subtract_background \
--parallax_correction \
--polarization_correction \
--solid_angle_correction \
--detector_absorption_correction \
--remove_bragg_peaks replace_by_median=True window_size_px=11
```
### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsdata.import``` step, which is the raw_data.dsdata.

- ```--fsave```:  
The new file to save after the ```dsimage.clean``` step, for example, the cleaned_data.dsdata.

- ```--read_dname```:  
The dataset name of image files to be preprocessed in ```--fname``` file. Usually the ```read_dname``` is ```image_file``` because you have imported your diffraction data by running ```dsdata.import --image_file /DATA/PATH/image_*.cbf```.

- ```--apply_detector_mask```:  
The first image preprocessing step, which will apply the user-defined detector mask, and then filter out pixels with intensity beyong range (value_vmin, value_vmax), or with radial positions beyond range (radius_rmin_px, radius_rmax_px) to the detector center. The default parameters are 
```
radius_rmin_px=40 radius_rmax_px=None value_vmin=0 value_vmax=10000
```
- ```--remove_bad_pixels```:  

- ```--subtract_background```:  

- ```--parallax_correction```:  

- ```--polarization_correction```:  

- ```--solid_angle_correction```:  

- ```--detector_absorption_correction```:  

- ```--remove_bragg_peaks```:  

