dsmap.operate
=====

### Full parameter command line:
```
dsmap.operate \
--fname ./data.dsdata \
--read_dname merge_volume \
--laue_symmetrization operator="+++,+-+,---,-+-" \
--friedel_symmetrization \
--background_subtraction contribute_window=5 expand_scale=4 \
--coordinate_conversion
```
### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsmap.merge```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-MERGE.md), the default is ```./data.dsdata```.

- ```--read_dname```:  
The dname of diffraction volume to be processed in ```--fname```. The default is ```merge_volume```.

- ```--laue_symmetrization```:
To perform Laue symmetrization to the diffraction volume. The default parameter is shown below, while users can set other operation based on the space group of their own protein.
```
  operator="+++,+-+,---,-+-"
```

- ```--friedel_symmetrization```:  
To perform Friedel symmetrization to the diffraction volume. No parameters are needed.

- ```--background_subtraction```:  
To perform the isotropic component subtraction. Basically no parameters are needed. The ```expand_scale``` is used to expand the volume lattice by ```expand_scale``` times for more accurate calculation of the isotropic component. Also, each voxel will contribute its intensity to ```contribute_window``` neighboring radial distance in order to get more accurate isotropic component. You don't need to specify these two parameters in general.
```
  contribute_window=5 expand_scale=4
```

- ```--coordinate_conversion```:  
This option is used for visualization only. The lattice vectors are not necessarily orthogonal, so we need to convert the HKL to XYZ lattice to visualize the diffraction volume.
