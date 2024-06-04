import argparse
import torch
import torch.nn as nn
import numpy as np
from arch import STASUNet
import os
import glob
import cv2
import datasets
import yaml
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity 
import lpips
from scipy.stats import norm
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test code')
parser.add_argument("--config", default='STASUNet.yml', type=str, help="training config file")
parser.add_argument('--resultDir', type=str, default='results', help='save output location')
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--retrain', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

resultDirOutImg = os.path.join(args.resultDir,"results")
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)
output_file = 'TestResults_' + args.resultDir + '.txt'

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)

if config.model.network=='STASUNet':
    model = STASUNet(num_in_ch=config.model.num_in_ch,num_out_ch=config.model.num_out_ch,num_feat=config.model.num_feat,num_frame=config.dataset.num_frames,deformable_groups=config.model.deformable_groups,num_extract_block=config.model.num_extract_block,
                    num_reconstruct_block=config.model.num_reconstruct_block,center_frame_idx=None,hr_in=config.model.hr_in,img_size=config.dataset.image_size,patch_size=config.model.patch_size,embed_dim=config.model.embed_dim, depths=config.model.depths,num_heads=config.model.num_heads,
                    window_size = config.model.window_size,patch_norm=config.model.patch_norm,final_upsample="Dual up-sample")
else:
    print("please specify model name!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '/home/edward/BVI-Mamba/results/best_model.pth.tar'
try:
    model.load_state_dict(torch.load(checkpoint_path),map_location=device)
except:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key[7:]] = checkpoint[key]  # Remove 'module.' prefix
            del checkpoint[key]
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# crop input size
overlapRatio = 1./4.
hpatch = config.dataset.image_size
wpatch = config.dataset.image_size
hgap = int(float(hpatch)*overlapRatio)
wgap = int(float(wpatch)*overlapRatio)
# Create weight for each patch
a = norm(hpatch/2, hpatch/6).pdf(np.arange(hpatch)) # gaussian weights along width
b = norm(wpatch/2, wpatch/6).pdf(np.arange(wpatch)) # gaussian weights along height
wmap = np.matmul(a[np.newaxis].T,b[np.newaxis]) # 2D weight map
wmap = wmap/wmap.sum()
# Repeat the 2D weight map along the third dimension
wmap = np.repeat(wmap[:, :, np.newaxis], 3, axis=2)

# =====================================================================

# Read folder names from the testing txt file
with open(config.dataset.val_file, 'r') as file:
    testfolder_names = file.read().splitlines()

dataset = datasets.__dict__[config.dataset.type](config)
train_data, val_data = dataset.load_lowlight(topatch=False)

lpips_model = lpips.LPIPS(net='alex') 
list_psnr = []
list_ssim = []
list_lpips = []

# initialize a counter
frame_counter = 0

for _count, sample in enumerate(tqdm(val_data, desc="Processing")):
    # from tensor to numpy
    img = sample['image']
    img = img.squeeze()
    img = img.cpu().numpy()

    frame_counter += 1
    
    # Run through each overlaping patch
    himg = img.shape[0]
    wimg = img.shape[1]

    weightMap = np.zeros((himg,wimg,3),np.float32)
    probMap = np.zeros((himg,wimg,3),np.float32)

    if frame_counter % 10 != 0:
        continue
    # print(f"calculate metrics for image {sample['img_id'][0]}")
    # stitch patches into frame
    for starty in np.concatenate((np.arange( 0, himg-hpatch, hgap),np.array([himg-hpatch])),axis=0):
        for startx in np.concatenate((np.arange( 0, wimg-wpatch, wgap),np.array([wimg-wpatch])),axis=0):
            crop_img = img[starty:starty+hpatch, startx:startx+wpatch]

            weightMap[starty:starty+hpatch, startx:startx+wpatch] += wmap
                
            # reshape and totensor
            image = crop_img
            if config.model.network=='STASUNet':
                image = image.transpose((3, 2, 0, 1))
            else:
                image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            if config.model.network=='STASUNet':
                image = (image-0.5)/0.5
            else:
                vallist = [0.5]*image.shape[0]
                normmid = transforms.Normalize(vallist, vallist)
                image = normmid(image)
            image = image.unsqueeze(0)

            inputs = image.to(device)
            with torch.no_grad():
                output = model(inputs)
                output = output.squeeze(0)
                output = output.cpu().numpy()
                output = output.transpose((1, 2, 0))

            probMap[starty:starty+hpatch, startx:startx+wpatch] += output*wmap

    # normalise weight
    probMap /= weightMap
    # clip to range [-1,1]
    probMap = np.clip(probMap, -1, 1)
    probMap = (probMap*0.5 + 0.5)*255

    # if int(subname[-1].split('.')[0]) < 10:  # only save the first 10 output frames for each scene 
    # cv2.imwrite(os.path.join(resultDirOutImg, sample['img_id'][0]+'.png'),probMap.astype(np.uint8))

    pred = probMap
    gt = sample['groundtruth'].squeeze(0)
    gt = gt.cpu().numpy()
    gt = gt.astype('float32')*255

    psnrvalue = peak_signal_noise_ratio (gt, pred, data_range=255)
    ssimvalue = structural_similarity(gt, pred, channel_axis=2, data_range=255, multichannel=True)
    pred_tensor = torch.tensor(pred.transpose((2, 0, 1)), dtype=torch.float32)
    gt_tensor = torch.tensor(pred.transpose((2, 0, 1)), dtype=torch.float32)
    lpipsvalue = lpips_model(pred_tensor, gt_tensor).item()
    
    list_psnr.append(psnrvalue)
    list_ssim.append(ssimvalue)
    list_lpips.append(lpipsvalue)
    tqdm.write('PSNR:{:.5f}, SSIM:{:.5f}, LPIPS:{:.5f}'.format(np.mean(list_psnr), np.mean(list_ssim), np.mean(list_lpips)))

# Calculate and print average PSNR and SSIM
print('Average PSNR:', np.mean(list_psnr))
print('Average SSIM:', np.mean(list_ssim))
print('Average LPIPS:', np.mean(list_lpips))

# Save the values to a text file
with open(output_file, 'a') as file:
    file.write(f'Average PSNR: {np.mean(list_psnr)}\n')
    file.write(f'Average SSIM: {np.mean(list_ssim)}\n')
    file.write(f'Average LPIPS: {np.mean(list_lpips)}\n')

print(f'Values saved to {output_file}')


   

