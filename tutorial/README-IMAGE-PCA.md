dsimage.pca
=====

### Full parameter command line:
```
dsimage.pca \
--fname ./data.diffuse \
--fsave None \
--read_dname image_file \
--save_dname None \
--per_image_multiply_scale scale_factor \
--radial_pca_subtraction num_pca=3
```
### Description of each input:  
- ```--fname```:  
The file name that contains your data files. It's usually the result of [```dsimage.scale```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-IMAGE-SCALE.md), the default is the ```./data.diffuse```.

- ```--fsave```:  
The new file to save the result of ```dsimage.pca```, for example, the ```cleaned_data_scaled_with_pca.dsdata```. The default is the same as ```--fname```.

- ```--read_dname```:  
The dname of image files to be processed in ```--fname```. The default ```--read_dname``` is ```image_file```.

- ```--save_dname```:
The dname to record the processed image files, which is the same as ```--read_dname``` in default. You can set other dnames, but then you will need to indicate the correct dname later for other steps such as [```dsmap.merge```](https://github.com/zhenwork/dspack/blob/main/tutorial/README-VOLUME-MERGE.md).

- ```--per_image_multiply_scale```:  
The dname of scale factor to use for image scaling, which is ```scale_factor``` in default.

- ```radial_pca_subtraction```:  
The radial profile variance removal step. The input parameter can specify the number of pca components to remove, the default number is ```num_pca=3```.
