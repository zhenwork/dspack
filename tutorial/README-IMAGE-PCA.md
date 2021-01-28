dsimage.pca
=====

### Full parameter command line:
```
dsimage.pca \
--fname clean_data_scaled.dsdata \
--fsave clean_data_scaled_with_pca.dsdata \
--read_dname image_file \
--save_dname None \
--per_image_multiply_scale scale_factor \
--radial_pca_subtraction num_pca=3
```
### Description of each input:  
- ```--fname```:  
The file name that contains all required data files. It is usually the result from the ```dsimage.scale``` step, which is the ```cleaned_data_scaled.dsdata```.

- ```--fsave```:  
The new file to save after the ```dsimage.pca``` step, for example, the ```cleaned_data_scaled_with_pca.dsdata```.

- ```--read_dname```:  
The dataset name of image files to be processed in ```--fname```. The default ```--read_dname``` is ```image_file```.

- ```--save_dname```:
The dataset name for the processed image files after ```dsimage.pca```, which is the same as ```--read_dname``` in default. You can set other dnames, but then you need to indicate the correct dname later for other steps such as ```dsmap.merge```.

- ```--per_image_multiply_scale```:  
The dname of scale factor to use for image scaling, which is ```scale_factor``` in default.

- ```radial_pca_subtraction```:  
The radial profile variance removal step. The input parameter is to specify the number of pca components to remove, the default number is ```num_pca=3```.
