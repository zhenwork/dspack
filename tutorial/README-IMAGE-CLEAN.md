dsimage.clean
=====

### Full parameter command line:
```
dsimage.clean \
--fname ./data.dsdata \
--fsave None \
--read_dname image_file \
--save_dname None \
--save_dir ./datadir \
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
The file that contains your imported data files. It's usually the result of the [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) step, the default is the ```./data.dsdata```.

- ```--fsave```:  
The file to save the result of ```dsimage.clean```, for example, the ```cleaned_data.dsdata```. The default value is the same as ```--fname```.

- ```--read_dname```:  
The dname of image files to be processed. Usually ```--read_dname``` is ```image_file``` because you have imported your diffraction patterns by running ```dsdata.import --image_file /PATH/TO/image_*.cbf```.

- ```--save_dname```:
The dname of the processed image files after ```dsimage.clean```, which is the same as ```--read_dname``` in default. However, users can choose to set a different dname such as ```process_image_files```, but you need to be careful to set the right ```--read_dname``` later for other steps.

- ```--save_dir```:
The intermediate data will be saved in this folder. The default is ```./datadir```.

- ```--apply_detector_mask```:  
The image preprocessing step (1), which will apply the user-defined detector mask, and then filter out pixels with intensity beyond range (```value_vmin```, ```value_vmax```), or with radial positions beyond range (```radius_rmin_px```, ```radius_rmax_px```) to the detector center. Usually ```radius_rmin_px``` means the beamstop. If you have imported the background diffraction patterns, this step will automatically be applied to the background patterns in the same manner. The default parameters are shown below,  
```
      radius_rmin_px=40 radius_rmax_px=None value_vmin=0 value_vmax=10000
```

- ```--remove_bad_pixels```:  
The image preprocessing step (2), which filters out each pixel with extreme intensity beyond ```sigma``` times the standard deviation to the mean intensity inside a square box with (```window_size_px```, ```window_size_px```) centered at that pixel. If you have imported the background diffraction patterns, this step will automatically be applied to the background patterns in the same manner. The default parameters are shown below, where currently only algorithm 2 is suggested.  
```
      algorithm=2 sigma=5 window_size_px=11
```

- ```--subtract_background```:  
The image preprocessing step (3), which will subtract the non-crystal background patterns from crystal diffraction patterns. It will cause errors if you don't have or import your background patterns, or if you have different numbers of crystal and background patterns. No input parameters are required.  

- ```--parallax_correction```:  
The image preprocessing step (4-1), which applies the parallax correction to pixel positions in each diffraction pattern. If enabled, this parallax correction step will automatically be performed before polarization, solid angle, and detector absorption corrections, because the corrected pixel positions will also affect the other pixel intensity corrections. No other parameters are required.  

- ```--polarization_correction```:  
The image preprocessing step (4-2), which applies the polarization correction to each diffraction pattern. Although the polarization factor is hard coded for the ICH dataset, users have the choice to set their own polarization factors to +1, -1 or fractional number between (-1,1) in the previous step of [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) by saving their polarization factors to the ```extra_json.js``` file. The [standard parameter keywords](https://github.com/zhenwork/dspack/blob/main/tutorial/README-STANDARD-KEYWORDS.md) for polarization factor is ```polarization_fr```. An example is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/example/extra_json.js). No other parameters are necessary for input.  

- ```--solid_angle_correction```:  
The image preprocessing step (4-3), which applies the solid angle correction to each diffraction pattern. No parameters are required.  

- ```--detector_absorption_correction```:  
The image preprocessing step (4-4), which applies the detector absorption correction to each diffraction pattern. This correction requires the detector thickness and detector absorption coefficients. Be sure to include these numbers in your experiment. An example is [here](https://github.com/zhenwork/dspack/blob/main/tutorial/example/extra_json.js). No other parameters are required.  

- ```--remove_bragg_peaks```:  
The image preprocessing step (5), which predicts Bragg peak positions and clean them either by replacing with median intensities or by filtering out them all. If ```replace_by_median``` is enabled, the replaced pixel value will be the median intensity inside a square box with size (```window_size_px```,```window_size_px```). Please note that all Bragg peaks will finally be filtered out before being merged into the 3D diffraction volume, replacing by median is just used to evaluate more accurate radial intensity profiles of each pattern, and for better visualizations. It's totally acceptable if you don't want to replace by median, and then all predicted Bragg peaks will be removed in the image preprocessing stage. The default parameters are shown below,  
```
      replace_by_median=True window_size_px=11
```
