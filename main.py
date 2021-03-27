import os
import numpy as np
import torch
import PIL.Image as Image
from model import ResNet
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--pyramid_height', type=int, default=3, help='pyramid_height ')
parser.add_argument('--save_every', type=int, default=200, help='iterations to save')
parser.add_argument('--img_path', type=str, default="images/input.jpg", help='path to input')
parser.add_argument('--gray_path', type=str, default="images/gray.jpg", help='path to gray image')
parser.add_argument('--mask_path', type=str, default="images/mask.png", help='path to mask')
parser.add_argument('--result_path', type=str, default="results/", help='path to saved_result')

def main(): 
    opt = parser.parse_args()
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    img = Image.open(opt.img_path)
    img_gray   = Image.open(opt.gray_path) 
    mask = 1. - np.array(Image.open(opt.mask_path))/ 255.
    mask = np.expand_dims(mask, 0)
    train(opt.pyramid_height, mask, img, img_gray,  torch.cuda.FloatTensor, opt)


def get_pyramid_image(img, w_n,  h_n):
    img = img.resize((w_n, h_n))
    img = np.array(img).transpose(2,0,1)/ 255. 
    return torch.from_numpy(img)[None, :]
    
def get_model(input_depth):
    net = ResNet(input_depth, 3, 8, 32, 3)
    return net
  
def set_pyramid_lr(pyramid_height):  
    if pyramid_height == 5:
        num_iter = [401, 401,401,401,401]
        lr = [0.04, 0.02,0.01,0.005,0.002]
    elif pyramid_height == 4:
        num_iter = [401,401,401,401]
        lr = [0.02,0.01,0.005,0.002]
    elif pyramid_height == 3:
        num_iter = [401,401,401]
        lr = [0.01,0.005,0.003]
    elif pyramid_height == 2:
        num_iter = [401, 401]
        lr = [0.01, 0.003]
    else:
        num_iter = [1001]
        lr = [0.01]    
    return num_iter, lr

def get_pyramids( mask, img, img_gray, pyramid_height, dtype):
    img_gray_var_pyr = []
    img_var_pyr = []
    mask_var_pyr = []    
    w, h  = img.size
    mask_var = torch.from_numpy(mask)[None, :]    
    for k in range(pyramid_height):
        w_n = int(w / (2 ** (pyramid_height-1-k))) 
        h_n = int(h / (2 ** (pyramid_height-1-k)))
        img_gray_var_pyr.append(get_pyramid_image(img_gray, w_n, h_n)[:,:1,:,:].type(dtype))
        img_var_pyr.append(get_pyramid_image(img, w_n, h_n).type(dtype))
        mask_var_pyr.append(mask_var.type(dtype))
        mask_var = (1 - torch.nn.functional.max_pool2d((1-mask_var), kernel_size=3, stride=2, padding=1))
    mask_var_pyr.reverse()
    return  mask_var_pyr, img_var_pyr, img_gray_var_pyr 

def train( pyramid_height, mask, img, img_gray, dtype, opt):
    # build pyramids 
    num_iter, pyramid_lr = set_pyramid_lr(pyramid_height)    
    mask_var_pyr, img_var_pyr, img_gray_var_pyr  = get_pyramids(mask, img, img_gray, pyramid_height, dtype)
    
    mse = torch.nn.MSELoss().type(dtype)
    input_depth = [1,4,4,4,4]
    
    for j in range(pyramid_height):
        net = get_model( input_depth=input_depth[j]).type(dtype)
        if j == 0:
            net_input = img_gray_var_pyr[j]
        else:
            out =  np.clip(out.detach().cpu().numpy()[0]*255,0,255).astype(np.uint8)
            out = Image.fromarray(out.transpose(1, 2, 0))
            out_resized = out.resize((img_gray_var_pyr[j].size(3),img_gray_var_pyr[j].size(2)), Image.LANCZOS)  
            out_resized_var = torch.from_numpy(np.array(out_resized).transpose(2,0,1)/ 255.)[None, :].type(dtype)
            net_input = torch.cat((img_gray_var_pyr[j], out_resized_var), 1)
      
        print('starting colorization. Scale %d' %j)
        optimizer = torch.optim.Adam(net.parameters(), lr=pyramid_lr[j])
        
        for i in range(num_iter[j]):
            optimizer.zero_grad()
            out = net(net_input)
            loss = mse(out * mask_var_pyr[j], img_var_pyr[j] * mask_var_pyr[j])
            loss.backward()
            optimizer.step()
            
            print ('iter: %05d    loss: %f' % (i, loss.item()), '\r', end='')
            if  i % opt.save_every == 0:
                out_ = np.transpose(out.detach().cpu().numpy()[0]*255, [1,2,0])
                out_ = Image.fromarray(out_.astype('uint8'), 'RGB')
                out_.save("%s/scale_%d_iter_%05d_out.jpg"%(opt.result_path, j, i)) 
            
            
if __name__ == "__main__":
    main()            
