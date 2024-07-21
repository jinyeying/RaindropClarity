# RaindropClarity
[ECCV2024] "Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal"

### Abstract
 Existing raindrop removal datasets have two shortcomings. First, they consist of images captured by cameras with a focus on the background, leading to the presence of blurry raindrops. To our knowledge, none of these datasets include images where the focus is specifically on raindrops, which results in a blurry background. Second, these datasets predominantly consist of daytime images, thereby lacking nighttime raindrop scenarios. Consequently, algorithms trained on these datasets may struggle to perform effectively in raindrop-focused or nighttime scenarios. The absence of datasets specifically designed for raindrop-focused and nighttime raindrops constrains research in this area. In this paper, we introduce a large-scale, real-world raindrop removal dataset called Raindrop Clarity. Raindrop Clarity comprises 15,186 high-quality pairs/triplets (raindrops, blur, and background) of images with raindrops and the corresponding clear background images. There are 5,442 daytime raindrop images and 9,744 nighttime raindrop images. Specifically, the 5,442 daytime images include 3,606 raindrop- and 1,836 background-focused images. While the 9,744 nighttime images contain 4,838 raindrop- and 4,906 background-focused images. Our dataset will enable the community to explore background-focused and raindrop-focused images, including challenges unique to daytime and nighttime conditions. 

## RaindropClarity Dataset
 |Day_Train|[Dropbox]()|[BaiduPan](https://pan.baidu.com/s/1-vwhYA7jEDPAYHlznhcHCA?pwd=j9ay) code:j9ay |
 |:-----------:| :-----------: | :-----------: |
 |Night_Train|[Dropbox]()|[BaiduPan](https://pan.baidu.com/s/13x6-UzqxaJG7tKv2WMyMuQ?pwd=hmsw) code:hmsw| 


## Pre-trained Models: [BaiduPan](https://pan.baidu.com/s/1tzJX--WD7YsYbpc9nGBQ0w?pwd=i3dg) code:i3dg and Results
| Model Name | Model Dropbox | Model BaiduPan | Results Dropbox | Results BaiduPan |
| :----: | :-----------: | :----------: |:---------------: |  :----------: |
| DiT | :-----------: | [BaiduPan](https://pan.baidu.com/s/15HhmzEEJvO9q-VJgz9PEuA?pwd=xziv) code:xziv|:---------------: |  :----------: | 


## Evaluation
```
python calculate_psnr_ssim_sid.py
```
please change `base_path`, `time_of_day`, `model_name` accordingly.

## Test
```
bash run_eval_diffusion_day.sh
```
```
bash run_eval_diffusion_night.sh
```
Inside script, please change `model_name` accordingly. 
```
CUDA_VISIBLE_DEVICES=7 python eval_diffusion_day_dit.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=6 python eval_diffusion_day_rdiffusion.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_restomer.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_uformer.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_onego.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_idt.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_icra.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_atgan.py --sid "$sid"
```

## Train
```
CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_64.yml --test_set Raindrop_DiT
```
please change `daytime_64.yml`,`daytime_128.yml`,`daytime_256.yml` according to `model_name` and `image_size`.
