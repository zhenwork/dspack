dsmap.operate
=====

### Full parameter command line:
```
dsmap.operate \
--fname clean_data_scaled_with_pca_map.dsdata \
--read_dname merge_volume \
--laue_symmetrization operator="+++,+-+,---,-+-" \
--friedel_symmetrization \
--background_subtraction contribute_window=5 expand_scale=4 \
--coordinate_conversion
```
### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsmap.merge``` step, which is the ```cleaned_data_scaled_with_pca_map.dsdata```.

- ```--read_dname```:  
The dataset name of diffraction volume to be processed in ```--fname```. The default ```--read_dname``` is ```merge_volume```.

- ```--laue_symmetrization```:
To perform the Laue symmetrization to the diffraction volume. The default parameter is shown below, while users can set other operation based on the space group of their own protein.
```
  operator="+++,+-+,---,-+-"
```

- ```--friedel_symmetrization```:  
To perform the Friedel symmetrization to the diffraction volume. No parameters are needed.

- ```--background_subtraction```:  
To perform the isotropic component subtraction. Basically no parameters are needed. The ```expand_scale``` is used to expand the volume lattice by times for more accurate isotropic component calculation. Also, each voxel intensity will contribute to ```contribute_window``` neighboring radius in order to get more accurate isotropic component. You don't need to specify these two parameters in general.
```
  contribute_window=5 expand_scale=4
```

- ```--coordinate_conversion```:  
This option is used for visualization only. The lattice vectors are not necessarly orthogonal, so we need to convert the HKL lattice to XYZ lattice to visualize the diffraction volume.
