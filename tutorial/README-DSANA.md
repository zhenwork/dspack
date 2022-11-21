# dsana.stats.lunus.llm

### Full parameter command line:
```
dsana.stats.lunus.llm \
--fname ./data.dsdata \
--lunus_data lunus_llm \
--launch_lunus_model b_factor_mode=three max_resolution_A=1.4 model=llm symop=-1
```

### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsmap.deploy.lunus.llm```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md), the default is ```./data.dsdata```.

- ```--lunus_data```:  
The exported files for LLM, which is the ```--lunus_data``` value in [```dsmap.deploy.lunus.llm```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md) step, the default is ```lunus_llm```.

- ```launch_lunus_model```:  
Parameters for the LLM model. The ```b_factor_mode``` has three basic cases: ```zero```, ```iso```, and ```aniso```. Another input ```three``` means to run the LLM model for all three different cases. The ```max_resolution_A``` means the resolution cutoff in Angstrom. Other information is seen [here](https://github.com/mewall/lunus/blob/master/scripts/refine_llm.py) in the Lunus software.

# dsana.stats.phenix.merge_stats

### Full parameter command line:
```
dsana.stats.phenix.merge_stats \
--fname ./data.dsdata \
--read_dname phenix_merge_stats \
--rmax_A 1.4 \
--nshells 15
```

### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsmap.deploy.phenix.merge_stats```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md).

- ```--read_dname```:  
The dname exported for PHENIX, which is the ```--save_dname``` value in [```dsmap.deploy.phenix.merge_stats```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-DEPLOY.md) step. The default is dname ```phenix_merge_stats```.

- ```--rmax_A```:
The maximum resolution cutoff for [phenix.merging_statistics](https://www.phenix-online.org/documentation/reference/unmerged_data.html) in Angstrom. The default is 1.4.

- ```--nshells```:  
The number of shells to divide for [phenix.merging_statistics](https://www.phenix-online.org/documentation/reference/unmerged_data.html). The default is 15.
