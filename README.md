# Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation ([CVPR 2022](https://cvpr2022.thecvf.com/))
[Video](https://www.youtube.com/watch?v=RoVu3hvvzGg) | [Paper](https://arxiv.org/abs/2202.13162)

![Teaser image](figures/teaser.jpg)

**Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation**<br>
[Shengqu Cai](https://primecai.github.io/), [Anton Obukhov](https://www.obukhov.ai/), [Dengxin Dai](https://vas.mpi-inf.mpg.de/dengxin/), [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html)

Abstract: *We propose a pipeline to generate Neural Radiance Fields (NeRF) of an object or a scene of a specific class, conditioned on a single input image. This is a challenging task, as training NeRF requires multiple views of the same scene, coupled with corresponding poses, which are hard to obtain. Our method is based on π-GAN, a generative model for unconditional 3D-aware image synthesis, which maps random latent codes to radiance fields of a class of objects. We jointly optimize (1) the π-GAN objective to utilize its high-fidelity 3D-aware generation and (2) a carefully designed reconstruction objective. The latter includes an encoder coupled with π-GAN generator to form an auto-encoder. Unlike previous few-shot NeRF approaches, our pipeline is unsupervised, capable of being trained with independent images without 3D, multi-view, or pose supervision. Applications of our pipeline include 3d avatar generation, object-centric novel view synthesis with a single input image, and 3d-aware super-resolution, to name a few.*




## Instructions
### Environment
We use pytorch 1.7.0 with CUDA 10.1. To build the environment, run:

```
pip install -r requirements.txt
```

### Download and pre-process datasets
For CelebA, download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and extract the `img_align_celeba` split.

For Carla, download from https://github.com/autonomousvision/graf.

For ShapeNet-SRN, download from https://github.com/sxyu/pixel-nerf and remove the additional layer, so that there are 3 folders `chairs_train`, `chairs_val` and `chairs_test` within `srn_chairs`. Instances should be directly within these three folders.

Copy `img_csv/CelebA_pos.csv` to /PATH_TO/img_align_celeba/.

Copy `srn_chairs_train.csv`, `srn_chairs_train_filted.csv`, `srn_chairs_val.csv`, `srn_chairs_val_filted.csv`, `srn_chairs_test.csv` and `srn_chairs_test_filted.csv` under `/PATH_TO/srn_chairs`.

### Visualizing
Render novel views of the given image:

`python render_video_from_img.py --path=/PATH_TO/checkpoint_train.pth 
                                   --output_dir=/PATH_TO_WRITE_TO/ 
                                   --img_path=/PATH_TO_IMAGE/ 
                                   --curriculum="celeba" or "carla" or "srnchairs"`


Render videos and create gifs for the three datasets:

CELEBA

`
python render_video_from_dataset.py --path PRETRAINED_MODEL_PATH 
                                    --output_dir OUTPUT_DIRECTORY 
                                    --curriculum "celeba" 
                                    --dataset_path "/PATH/TO/img_align_celeba/" 
                                    --trajectory "front"
`

CARLA

`python render_video_from_dataset.py --path PRETRAINED_MODEL_PATH 
                                     --output_dir OUTPUT_DIRECTORY 
                                     --curriculum "carla" 
                                     --dataset_path "/PATH/TO/carla/*.png" 
                                     --trajectory "orbit"`

SRNCHAIRS

`python render_video_from_dataset.py --path PRETRAINED_MODEL_PATH 
                                     --output_dir OUTPUT_DIRECTORY 
                                     --curriculum "srnchairs" 
                                     --dataset_path "/PATH/TO/srn_chairs/" 
                                     --trajectory "orbit"`

### Linear interpolation
Render images and a video interpolating between 2 images.

`python linear_interpolation --path=/PATH_TO/checkpoint_train.pth --output_dir=/PATH_TO_WRITE_TO/`

### Hybrid Optimization
Since our model is feed-forward and uses a relatively compact latent codes, it most likely will not perform that well on yourself/very familiar faces---the details are very challenging to be fully captured by a single pass. Therefore, we provide a script performing hybrid optimization: predict a latent code using our model, then perform latent optimization as introduced in pi-GAN. The command to use is:

`
python --path PRETRAINED_MODEL_PATH --output_dir OUTPUT_DIRECTORY --curriculum ["celeba" or "carla" or "srnchairs"] --img_path /PATH_TO_IMAGE_TO_OPTIMIZE/   
`
Note that compare with vanilla pi-GAN inversion, we need significantly less iterations.

## Pretrained model
We provide pretrained model checkpoint files for the three datasets. Download from https://www.dropbox.com/s/lcko0wl8rs4k5qq/pretrained_models.zip?dl=0 and unzip to use.

## Citation

```
@inproceedings{cai2022pix2nerf,
  title={Pix2NeRF: Unsupervised Conditional p-GAN for Single Image to Neural Radiance Fields Translation},
  author={Cai, Shengqu and Obukhov, Anton and Dai, Dengxin and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3981--3990},
  year={2022}
}
```

## Acknowledgements
The code repo is built upon https://github.com/marcoamonteiro/pi-GAN. We thank the authors for releasing the code and providing support throughout the development of this project.
