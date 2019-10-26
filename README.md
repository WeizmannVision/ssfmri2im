# ssfmri2im
Code for NeurIPS 2019 paper "From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI"


Paper: https://arxiv.org/abs/1907.02431
Project page: http://www.wisdom.weizmann.ac.il/~vision/ssfmri2im/
video:

 [![ssfmri2im](http://img.youtube.com/vi/h2JhDAdaa-Q/0.jpg)](http://www.youtube.com/watch?v=h2JhDAdaa-Q "Self-supervision in natural-image reconstruction from fMRI")


----------
If you find our work useful in your research or publication, please cite our work:

```
@article{beliy2019voxels,
  title={From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI},
  author={Beliy, Roman and Gaziv, Guy and Hoogi, Assaf and Strappini, Francesca and Golan, Tal and Irani, Michal},
  journal={arXiv preprint arXiv:1907.02431},
  year={2019}
}
```
----------
##Basic usage:
1. Download "Generic Object Decoding" dataset (by Kamitani Lab)
```
http://brainliner.jp/data/brainliner/Generic_Object_Decoding
```

2. Download the images used in the experiment
```
http://image-net.org/download
```
For me it was easiest to download the relevant winds

more instructions here:
```
https://github.com/KamitaniLab/GenericObjectDecoding
```
3. Change Paths in config_file.py to match your file locations, specifically:
   - imagenet_wind_dir - point to the Imagenet image directory
   - external_images_dir - external iamges to be used in training, we use the Imagenet(2011) validation images
   - kamitani_data_mat - mat file containing the fMRI activations
4. Run run file, this will do the following:
   - create a NPZ file with the images used in the experiment.
   - Train an Encoder model and save it's weights
   - Train the full model and save the reconstructed images

example output:
![sketch](/collage.jpeg)




