# dsmap.deploy.lunus.llm

### Full parameter command line:
```
dsmap.deploy.lunus.llm \
--fname clean_data_scaled_with_pca_map.dsdata \
--read_dname merge_volume \
--pdb_dname pdb_refmac5 \
--lunus_data lunus_llm_refmac5
```

### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsmap.operate``` step, which is the ```cleaned_data_scaled_with_pca_map.dsdata```.

- ```--read_dname```:  
The dataset name of diffraction volume to be processed in ```--fname```. The default ```--read_dname``` is ```merge_volume```.

- ```--pdb_dname```:
The dname of imported PDB files. You may import multiple PDB files and save them as different pdb dnames with the ```dsdata.import``` method, such as the phenix-refined PDB and refmac5-refined PDB files. You need to specify which one to use for the LLM model.

- ```--lunus_data```:  
The exported files for LLM in Lunus will be saved in ```--lunus_data```, so you just need to specify ```--lunus_data``` when you are running the LLM model later in ```dsana.stats.lunus.llm```.


# dsmap.deploy.phenix.merge_stats

### Full parameter command line:
```
dsmap.deploy.phenix.merge_stats \
--fname clean_data_scaled_with_pca_map.dsdata \
--read_dname merge_volume \
--save_dname phenix_merge_stats \
--pdb_dname pdb_refmac5 \
--vmin -100 \
--vmax 1000 
```

### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsmap.operate``` step, which is the ```cleaned_data_scaled_with_pca_map.dsdata```.

- ```--read_dname```:  
The dataset name of diffraction volume to be processed in ```--fname```. The default ```--read_dname``` is ```merge_volume```.

- ```--pdb_dname```:
The dname of imported PDB files. You may import multiple PDB files and save them as different pdb dnames with the ```dsdata.import``` method, such as the phenix-refined PDB and refmac5-refined PDB files. You need to specify which one to use for phenix.merging_statistics.

- ```--save_dname```:  
The exported files for phenix.merging_statistics will be saved in ```--save_dname```, so you just need to specify ```--save_dname``` when you are running the PHENIX model later in ```dsana.stats.phenix.merge_stats```.

- ```--vmin```:  
The minimum voxel intensity to keep for running CChalf. The default number is -100.

- ```--vmax```:  
The maximum voxel intensity to keep for running CChalf. The default number is 1000.
