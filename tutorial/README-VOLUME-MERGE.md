dsmap.merge
=====

### Full parameter command line:
```
dsmap.merge \
--fname clean_data_scaled_with_pca.dsdata \
--fsave clean_data_scaled_with_pca_map.dsdata \
--read_dname image_file \
--save_dname merge_volume \
--per_image_multiply_scale scale_factor \
--select_image_idx None \
--merge_to_volume volume_center_vx=60 volume_size_vx=121 oversample=1
```
### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsimage.pca``` step, which is the ```cleaned_data_scaled_with_pca.dsdata```.

- ```--fsave```:  
The saved diffraction volume file after the ```dsmap.merge``` step, for example, the ```cleaned_data_scaled_with_pca_map.dsdata```.

- ```--read_dname```:  
The dataset name of image files to be merged in ```--fname```. The default ```--read_dname``` is ```image_file```.

- ```--save_dname```:
The dataset name for the merged diffraction volume file after ```dsmap.merge```, which is ```merge_volume``` in default.

- ```--per_image_multiply_scale```:  
The dname of scale factor to use for image scaling, which is ```scale_factor``` in default.

- ```--select_image_idx```:  
The selected image to merge into a volume, for example, ```--select_image_idx 50-100,200-300``` will merge images with index from ```[50,100]``` and ```[200,300]```, inclusively. 

- ```--merge_to_volume```:  
To turn on the merging algorithm. The diffraction patterns will be merged into a volume with size (```volume_size_vx```,```volume_size_vx```,```volume_size_vx```). The center of the volume will be (```volume_center_vx```,```volume_center_vx```,```volume_center_vx```), and the oversampling rate to Miller indices is ```oversample```. The default parameters are
```
  volume_center_vx=60 volume_size_vx=121 oversample=1
```
which means that each index of H, K, and L ranges from -60 to 60. However, for visualization, we use a larger volume with parameters as follows,
```
  volume_center_vx=180 volume_size_vx=361 oversample=3
```
