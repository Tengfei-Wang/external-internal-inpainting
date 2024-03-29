# EII: Image Inpainting with External-Internal Learning and Monochromic Bottleneck
> Image Inpainting with External-Internal Learning and Monochromic Bottleneck   
> Tengfei Wang*, Hao Ouyang*, Qifeng Chen   
> CVPR 2021   

[paper](https://arxiv.org/abs/2104.09068) | [project website](https://tengfei-wang.github.io/EII/index.html ) | [video](https://youtu.be/JFBEL7qPFVc)

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
### Colorization
To try our internal colorization method:
```
python main.py  --img_path images/input2.png --gray_path images/gray2.png  --mask_path images/mask2.png  --pyramid_height 3
```
The colorization results are placed in ./results.  
In case the colorization results are unsatisfactory, you may consider changing the pyramid_height (2~5 work well for most cases).

### Reconstruction
For the monochromic reconstruction stage, multiple inpainting networks can be applied as backbones by modifying the original input image, like:
```
input_new = torch.concat([input_RGB, input_gray],1) #input_new is 4-channel
output = backbone_model(input_new, mask) #output is single-channel
loss = criterion(output, input_gray)
```

## Citation
If you find this work useful for your research, please cite:
``` 
@inproceedings{wang2021image,
  title={Image Inpainting with External-internal Learning and Monochromic Bottleneck},
  author={Wang, Tengfei and Ouyang, Hao and Chen, Qifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5120--5129},
  year={2021}
}
```


## Contact
Please send emails to tengfeiwang12@gmail.com or ououkenneth@gmail.com if there is any question

##  Acknowledgement
We thank the authors of [DIP](https://github.com/DmitryUlyanov/deep-image-prior) for sharing their codes.
