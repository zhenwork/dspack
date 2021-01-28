dsimage.scale
====

### Full parameter command line:
```
dsimage.scale \
--fname cleaned_data.dsdata \
--fsave cleaned_data_scaled.dsdata \
--read_dname image_file \
--save_dname scale_factor \
--scale_by_radial_profile radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4 \
--scale_by_water_ring_intensity radius_rmin_px=None radius_rmax_px=None radius_rmin_A=5 radius_rmax_A=1.82 \
--scale_by_overall_intensity radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4 \
--scale_by_bragg_intensity
(select one scale factor only)
```
There are four lines with four different scale factors, but you can only enable one of them, and delete other three lines. For computational efficiency, the scale factor of the dataset will be saved as a numpy array and recorded as dname=```scale_factor``` in ```--fsave```, rather than be multipled to each image because saving images is slow. Other steps such as ```dsimage.pca``` or ```dsmap.merge``` will automatically look for the dname of ```scale_factor``` and read the scale factor out.

### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsimage.clean``` step, which is the ```cleaned_data.dsdata```.

- ```--fsave```:  
The new file to save after the ```dsimage.scale``` step, for example, the ```cleaned_data_scaled.dsdata```.

- ```--read_dname```:  
The dataset name of image files to be scaled in ```--fname```. Usually the ```read_dname``` is ```image_file``` because the default ```--save_dname``` is ```image_file``` in the previous step ```dsimage.clean```.

- ```--save_dname```:
The dataset name for the calculated scale factor after ```dsimage.scale```, which is ```scale_factor``` as default. You can set other dnames, but then you need to indicate the correct dname later for other steps such as ```dsimage.pca```.

- ```--scale_by_radial_profile```:  
The scaling method (1), which uses the image radial intensity profile to scale each image, the default parameters to specify the resolution range of the radial profle are shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4
```

- ```--scale_by_water_ring_intensity```:  
The scaling method (2), which uses the average pixel intensity inside the isotropic ring, the default parameters to specify the resolution range of the isotropic ring are shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=5 radius_rmax_A=1.82
```

- ```--scale_by_overall_intensity```:  
The scaling method (3), which uses the average pixel intensity inside the whole resolution range, the default parameters to specify the resolution range are shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4
```

- ```--scale_by_bragg_intensity```:  
The scaling method (3), which uses the Bragg intensity scale factor to scale each image. No parameters are needed but be sure to run ```dsdata.import``` to import your Bragg scale factor file, for example, the dials report file as mentioned [here](https://github.com/zhenwork/dspack/edit/main/tutorial/README-DATA-IMPORT.md).
