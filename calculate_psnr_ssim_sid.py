import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim
import torch
import lpips
# loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')

base_path = '/home1/yeying/release/RDiffusion/results/NightRaindrop/'
time_of_day = 'night' 
# base_path = '/home1/yeying/release/RDiffusion/results/RainDrop/'
# time_of_day = 'day' 

dayend_path   = 'DayRainDrop/gt/'
nightend_path = 'NightRainDrop/gt/'

#model_name   = 'Raindrop_DiT/'
#model_name   = 'Uformer/'
model_name   = 'restormer/'

#model_name   = 'RDiffusion/'
#model_name   = 'IDT/'
#model_name   = 'onego/'
#model_name   = 'ICRA256/'
#model_name   = 'atgan/'

def generate_gt_path(time_of_day):
    if time_of_day == 'day':
        gt_path = base_path + model_name + dayend_path
    elif time_of_day == 'night':
        gt_path = base_path + model_name + nightend_path
    return gt_path

gt_path = generate_gt_path(time_of_day)
results_path = gt_path.replace('gt', 'output')
print(results_path)

imgsName = sorted(os.listdir(results_path))
imgslist = []
for i in range(len(imgsName)):
    path_1 = os.path.join(results_path, imgsName[i])
    dir_1 = sorted(os.listdir(path_1))
    for j in range(len(dir_1)):
        imgslist.append(path_1 + '/' +dir_1[j])
# gtsName = sorted(os.listdir(gt_path))
print('-len(imgslist)-',len(imgslist))
gtslist = []
for i in range(len(imgslist)):
    gts = imgslist[i].replace(results_path,gt_path)
    # gts = gts.replace('_output.','_gt.')
    gtslist.append(gts)
print(gtslist[0])
print(gtslist[-1])
print(imgslist[0])
print(imgslist[-1])
cumulative_psnr, cumulative_ssim,cumulative_lpips = 0, 0, 0

for i in range(len(imgslist)):
    # print(imgslist[i])
    res = cv2.imread(imgslist[i], cv2.IMREAD_COLOR)
    gt  = cv2.imread(gtslist[i], cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)

    torchres = torch.from_numpy(res.transpose((2, 0, 1))).float().unsqueeze(0)
    torchgt  = torch.from_numpy(gt.transpose((2, 0, 1))).float().unsqueeze(0)
    # torchres = torchres/255.0 *2 - 1
    # torchgt = torchgt/255.0 *2 - 1
    # print('-torchres-',torchres.shape,'-torchgt-',torchgt.shape)
    cur_lpips = loss_fn_vgg(torchres, torchgt)
    # print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    cumulative_lpips +=cur_lpips.cpu().data.numpy()[0][0][0][0]
    if i%100==0:
        print('Testing set,'+str(i)+' PSNR is %.4f, SSIM is %.4f and lpips is %.4f' % (cumulative_psnr / (i+1), cumulative_ssim / (i+1), cumulative_lpips / (i+1)))
print(time_of_day)
print('%s, PSNR is %.4f, SSIM is %.4f and lpips is %.4f' % (model_name, cumulative_psnr / len(imgslist), cumulative_ssim / len(imgslist), cumulative_lpips / len(imgslist)))

#night
#input         PSNR is 24.7756, SSIM is 0.7263 and lpips is 0.2091
#Raindrop_DiT, PSNR is 26.2295, SSIM is 0.8263 and lpips is 0.1106
#Uformer,      PSNR is 25.2648, SSIM is 0.8158 and lpips is 0.1419
#restormer,    PSNR is 26.8733, SSIM is 0.8479 and lpips is 0.1233 

#RDiffusion,   PSNR is 26.4834, SSIM is 0.8310 and lpips is 0.1121
#IDT,          PSNR is 26.8061, SSIM is 0.8512 and lpips is 0.1252 
#onego,        PSNR is 25.4149, SSIM is 0.8204 and lpips is 0.1391
#ICRA256,      PSNR is 24.5673, SSIM is 0.7874 and lpips is 0.1818 
#atgan,        PSNR is 24.3764, SSIM is 0.7734 and lpips is 0.1852

#################################################################
#day           PSNR is 21.9165, SSIM is 0.5604 and lpips is 0.2472
#Raindrop_DiT, PSNR is 26.0286, SSIM is 0.7515 and lpips is 0.1061
#Uformer,      PSNR is 25.7826, SSIM is 0.7274 and lpips is 0.1478
#restormer,    PSNR is 26.0863, SSIM is 0.7482 and lpips is 0.1311 

#RDiffusion,   PSNR is 25.5225, SSIM is 0.7344 and lpips is 0.1110
#IDT,          PSNR is 26.0496, SSIM is 0.7364 and lpips is 0.1408
#onego,        PSNR is 25.3921, SSIM is 0.7166 and lpips is 0.1472
#ICRA256,      PSNR is 23.8187, SSIM is 0.6704 and lpips is 0.1913
#atgan,        PSNR is 23.6169, SSIM is 0.6581 and lpips is 0.1997