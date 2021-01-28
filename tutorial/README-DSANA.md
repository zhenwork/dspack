# dsana.stats.lunus.llm

### Full parameter command line:
```
dsana.stats.lunus.llm \
--fname clean_data_scaled_with_pca_map.dsdata \
--lunus_data lunus_llm_refmac5 \
--launch_lunus_model b_factor_mode=three max_resolution_A=1.4 model=llm symop=-1
```

### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsmap.deploy.lunus.llm``` step, which is the ```cleaned_data_scaled_with_pca_map.dsdata```.

- ```--lunus_data```:  
The exported files for LLM, which is the ```--lunus_data``` key word in the [```dsmap.deploy.lunus.llm```](https://github.com/zhenwork/dspack/edit/main/tutorial/README-DEPLOY.md) step, the default is ```lunus_llm_refmac5```.

- ```launch_lunus_model```:  
Parameters for the LLM model. The ```b_factor_mode``` has three basic cases: ```zero```, ```iso```, and ```aniso```. Another input value ```three``` means to calculate the LLM model for all three different cases. The ```max_resolution_A``` means the resolution cutoff in Angstrom. Other information is seen [here](https://github.com/mewall/lunus/blob/master/scripts/refine_llm.py) in the Lunus software.

# dsana.stats.phenix.merge_stats

### Full parameter command line:
```
	dsana.stats.phenix.merge_stats \
	--fname clean_data_scaled_with_pca_map.dsdata \
  --read_dname phenix_merge_stats \
  --rmax_A 1.4 \
  --nshells 15
```

### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsmap.deploy.phenix.merge_stats``` step.

- ```--read_dname```:  
The dataset name exported for PHENIX, which is the input of ```--save_dname``` for [```dsmap.deploy.phenix.merge_stats```](https://github.com/zhenwork/dspack/edit/main/tutorial/README-DEPLOY.md). The default is dname ```phenix_merge_stats```.

- ```--rmax_A```:
The maximum resolution cutoff for phenix.merging_statistics in Angstrom. The default is 1.4.

- ```--nshells```:  
The number of shells to divide for phenix.merging_statistics. The default is 15.
