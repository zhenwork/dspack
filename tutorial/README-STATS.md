dsana.stats.cc.map
=====

### Full parameter command line:
```
dsana.stats.cc.map \
--fname clean_data_scaled_with_pca_map_1.dsdata clean_data_scaled_with_pca_map_2.dsdata clean_data_scaled_with_pca_map_3.dsdata \
--fsave cc_map.out \
--read_dname merge_volume \
--pdb_dname pdb_refmac5 \
--nshells 15 \
--vmin -100 \
--vmax 1000 \
--rmax_A 1.4 \
--rmin_A 50 
```
Note that the comparsion is performed for the symmetrized anisotropic diffuse maps only.

### Description of each input:  
- ```--fname```:  
A list of files to compare, such as ```clean_data_scaled_with_pca_map.dsdata``` files from multiple datasets or various processing choices.

- ```--fsave```:  
The CC statistics for N files will be kept in a N\*N matrix, and saved to ```--fsave``` file as a numpy array.

- ```--read_dname```:
The dname to compare in ```--fname``` files, the default is ```merge_volume```.

- ```pdb_dname```:  
The dname of PDB file for CC calculation, the default is ```pdb_file```.

- ```--nshells```:  
The number of shells to divide for the calculation of resolution-dependent CC. The default is 15.

- ```vmin```:  
The minimum voxel value to keep for CC calculation. The default is -100.

- ```vmax```:  
The minimum voxel value to keep for CC calculation. The default is 1000.

- ```rmax_A```:  
The maximum resolution cutoff for CC calculation in Angstrom. The default is 1.4.

- ```rmin_A```:  
The minimum resolution cutoff for CC calculation in Angstrom. The default is 50.
