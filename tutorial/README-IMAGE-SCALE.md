dsimage.scale
====

### Full parameter command line:
```
dsimage.scale \
--fname ./data.diffuse \
--fsave None \
--read_dname image_file \
--save_dname scale_factor \
--scale_by_radial_profile radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4 \
--scale_by_water_ring_intensity radius_rmin_px=None radius_rmax_px=None radius_rmin_A=5 radius_rmax_A=1.82 \
--scale_by_overall_intensity radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4 \
--scale_by_bragg_intensity
(select one scale factor only)
```
There are four lines with four different scale factors, but you can only enable one of them, and delete other three lines. For computational efficiency, the scale factor of a dataset will be saved as a numpy array and recorded to ```--save_dname``` in file ```--fsave```, rather than be multiplied to each image because saving new images is slow. Other steps such as [```dsimage.pca```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-PCA.md) or [```dsmap.merge```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-MERGE.md) will automatically look for ```--save_dname``` to read out the scale factor.

### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It is usually the result of [```dsimage.clean```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-CLEAN.md), the default is ```./data.diffuse```.

- ```--fsave```:  
The file name to save the result of ```dsimage.scale```, for example, you can use ```cleaned_data_scaled.dsdata``` or any other file names. The default is the same as ```--fname``.

- ```--read_dname```:  
The dname of diffraction image files to be scaled in ```--fname```. Usually the ```--read_dname``` is ```image_file``` because the default ```--save_dname``` is ```image_file``` in the previous step [```dsimage.clean```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-CLEAN.md).

- ```--save_dname```:
The dname to save the calculated scale factor, which is ```scale_factor``` in default. You can use other dnames to record the scale factor, but you will need to indicate this dname later in other steps that require the scale factor, such as [```dsimage.pca```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-PCA.md).

- ```--scale_by_radial_profile```:  
The scaling method (1), which uses the image radial intensity profile to scale each image, and the default resolution range of the radial profile is shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4
```

- ```--scale_by_water_ring_intensity```:  
The scaling method (2), which uses the average pixel intensity inside the isotropic ring, and the default resolution range of the isotropic ring is shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=5 radius_rmax_A=1.82
```

- ```--scale_by_overall_intensity```:  
The scaling method (3), which uses the average pixel intensity inside the whole resolution range, and the default resolution range is shown below.  
```
    radius_rmin_px=None radius_rmax_px=None radius_rmin_A=50 radius_rmax_A=1.4
```

- ```--scale_by_bragg_intensity```:  
The scaling method (4), which uses the Bragg intensity scale factor to scale each image. No parameters are needed but be sure to use [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) to import your file with the Bragg scale factor, for example, the dials report file as mentioned [here](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md).
