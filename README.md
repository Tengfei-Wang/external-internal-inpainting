# Image Inpainting with External-internal Learning and Monochromic Bottleneck
This repository is for the CVPR 2021 paper: 'Image Inpainting with External-internal Learning and Monochromic Bottleneck'

[paper](https://arxiv.org/abs/2104.09068) | [project website](https://tengfei-wang.github.io/EII/index.html )

## Introduction
The proposed method can be  applied to improve the color consistency of leaning-based image inpainting results.   The progressive internal color propagation  shows strong performance even with large mask ratios. 
<img src="pics/color.jpg" height="305px"/>
<img src="pics/multi-ratio.jpg" height="360px"/>
## Prerequisites
- Python 3.6
- Pytorch 1.6
- Numpy

## Installation
```
git clone https://github.com/Tengfei-Wang/external-internal-inpainting.git
cd external-internal-inpainting
```

## Quick Start 
To try our internal colorization method:
```
python main.py  --img_path images/input2.png --gray_path images/gray2.png  --mask_path images/mask2.png  --pyramid_height 3
```
The colorization results are placed in ./results.

For the monochromic reconstruction stage, multiple inpainting networks can be applied as backbones by modifying the original input image, like:
```
input_new = torch.concat([input_RGB, input_gray],1) #input_new is 4-channel
output = backbone_model(input_new, mask) #output is single-channel
loss = criterion(output, input_gray)
```

## Citation
If you find this work useful for your research, please cite:
``` 
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Tengfei and Ouyang, Hao and Chen, Qifeng},
    title     = {Image Inpainting With External-Internal Learning and Monochromic Bottleneck},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {5120-5129}
}
```


## Contact
Please send emails to tengfeiwang12@gmail.com  if there is any question
