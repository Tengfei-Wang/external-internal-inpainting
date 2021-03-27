# Image Inpainting with External-internal Learning and Monochromic Bottleneck
This repository is for the CVPR 2021 paper: 'Image Inpainting with External-internal Learning and Monochromic Bottleneck'

[paper]() | [project website]( )

## Introduction
The proposed method can be  applied to improve the color consistency of leaning-based image inpainting results.   The progressive internal color propagation  shows strong performance even with large mask ratios. 
<img src="pics/color.jpg" height="220px"/>
<img src="pics/multi-ratio.jpg" height="260px"/>
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

## Citation
If you find this work useful for your research, please cite:
```
             
```


## Contact
Please send email to tengfeiwang12@gmail.com  if there is any question
