dsmap.merge
=====

### Full parameter command line:
```
dsmap.merge \
--fname ./data.dsdata \
--fsave None \
--read_dname image_file \
--save_dname merge_volume \
--per_image_multiply_scale scale_factor \
--select_image_idx None \
--merge_to_volume volume_center_vx=60 volume_size_vx=121 oversample=1
```
### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsimage.pca```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-PCA.md), the default is ```./data.dsdata```.

- ```--fsave```:  
The file name to save the result of ```dsmap.merge```, the default is the same as ```--fname```.

- ```--read_dname```:  
The dname of image files to be merged in ```--fname```. The default is ```image_file```.

- ```--save_dname```:
The dname to record the merged diffraction volume for ```dsmap.merge```, and the default is ```merge_volume```.

- ```--per_image_multiply_scale```:  
The dname of scale factor to use for image scaling, the default is ```scale_factor```.

- ```--select_image_idx```:  
The selected image index to merge, for example, ```--select_image_idx 50-100,200-300``` means merging images with index from ```[50,100]``` and ```[200,300]```, inclusively. The default is None (meaning all images).

- ```--merge_to_volume```:  
To turn on the merging algorithm. The diffraction patterns will be merged into a 3D volume with size (```volume_size_vx```,```volume_size_vx```,```volume_size_vx```). The center of the volume is (```volume_center_vx```,```volume_center_vx```,```volume_center_vx```), and the oversampling rate to Miller indices is ```oversample```. The default parameters are
```
  volume_center_vx=60 volume_size_vx=121 oversample=1
```
which means that each index of H, K, and L ranges from -60 to 60. However, for improved visualization, we use a larger volume with parameters as follows,
```
  volume_center_vx=180 volume_size_vx=361 oversample=3
```
