# dsmap.deploy.lunus.llm

### Full parameter command line:
```
dsmap.deploy.lunus.llm \
--fname ./data.dsdata \
--read_dname merge_volume \
--pdb_dname pdb_file \
--lunus_data lunus_llm
```

### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsmap.operate```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-OPERATE.md), the default is ```./data.dsdata```.

- ```--read_dname```:  
The dname of diffraction volume to be processed in ```--fname```. The default is ```merge_volume```.

- ```--pdb_dname```:
The dname of PDB file to use. You may import multiple PDB files and save them as different pdb dnames in the [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) step, such as the phenix-refined and refmac5-refined PDB files. You need to specify which one to use for the LLM model. The default is ```pdb_file```.

- ```--lunus_data```:  
The exported files for LLM will be saved in ```--lunus_data```, so you only need to specify ```--lunus_data``` next time when you are running the LLM model with [```dsana.stats.lunus.llm```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DSANA.md). The default is ```lunus_llm```.


# dsmap.deploy.phenix.merge_stats

### Full parameter command line:
```
dsmap.deploy.phenix.merge_stats \
--fname ./data.dsdata \
--read_dname merge_volume \
--save_dname phenix_merge_stats \
--pdb_dname pdb_file \
--vmin -100 \
--vmax 1000 
```

### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsmap.operate```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-OPERATE.md), the default is ```./data.dsdata```.

- ```--read_dname```:  
The dname of diffraction volume to be processed in ```--fname```. The default is ```merge_volume```.

- ```--pdb_dname```:
The dname of imported PDB file. You may import multiple PDB files and save them as different pdb dnames  in the [```dsdata.import```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DATA-IMPORT.md) step, such as the phenix-refined and refmac5-refined PDB files. You need to specify which one to use for [phenix.merging_statistics](https://www.phenix-online.org/documentation/reference/unmerged_data.html). The default is ```pdb_file```.

- ```--save_dname```:  
The exported files for [phenix.merging_statistics](https://www.phenix-online.org/documentation/reference/unmerged_data.html) will be saved in ```--save_dname```, so that you only need to specify ```--save_dname``` next time when you are running the PHENIX model later in [```dsana.stats.phenix.merge_stats```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DSANA.md). The default is ```phenix_merge_stats```.

- ```--vmin```:  
The minimum voxel intensity to keep for running CChalf. The default number is -100.

- ```--vmax```:  
The maximum voxel intensity to keep for running CChalf. The default number is 1000.
