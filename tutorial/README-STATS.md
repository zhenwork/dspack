dsana.stats.cc.map
=====

### Full parameter command line:
```
dsana.stats.cc.map \
--fname map_1.dsdata map_2.dsdata map_3.dsdata \
--fsave dsana.stats.cc.map.out \
--read_dname merge_volume \
--pdb_dname pdb_file \
--nshells 15 \
--vmin -100 \
--vmax 1000 \
--rmax_A 1.4 \
--rmin_A 50 
```
Note that the comparison is performed for the symmetrized anisotropic diffuse maps only.

### Description of each input:  
- ```--fname```:  
A list of files to compare, such as diffuse data files after the [```dsmap.operate```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-OPERATE.md) step for multiple datasets or various processing choices.

- ```--fsave```:  
The CC statistics for N files will be kept in a N\*N matrix, and saved to ```--fsave``` file as a numpy array. The default is ```dsana.stats.cc.map.out```.

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
