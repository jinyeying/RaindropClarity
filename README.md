# RaindropClarity (ECCV'2024) (CVPR-NTIRE'2025)

## Introduction
> [Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](https://arxiv.org/abs/2407.16957)<br>
> European Conference on Computer Vision (ECCV'2024) (CVPR-NTIRE'2025)
> 
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.16957)
[[Poster]](https://github.com/jinyeying/RaindropClarity/blob/main/poster_slides/RaindropClarity_poster.pdf) 
[[Slides]](https://github.com/jinyeying/RaindropClarity/blob/main/poster_slides/RaindropClarity_PPT.pdf) 
[[Video]](https://www.youtube.com/watch?v=LSGvCuT46XU)

# Workshop and Challenges @ CVPR-NTIRE'2025
🔥 The first challenge on Day and Night Raindrop Removal for Dual-Focused Images<br>
[[Competitions]](https://codalab.lisn.upsaclay.fr/competitions/21345) [[ChallengePage]](https://lixinustc.github.io/CVPR-NTIRE2025-RainDrop-Competition.github.io/)

## RaindropClarity Train Data (challenge and ECCV'2024)
 |Day_Train   |[Dropbox](https://www.dropbox.com/scl/fi/qes7r934c10qzb21funoj/DayRainDrop_Train.zip?rlkey=bdqa53wgvmhj9x1yf40q0c1p7&st=4taffjkx&dl=0) | [BaiduPan](https://pan.baidu.com/s/1-vwhYA7jEDPAYHlznhcHCA?pwd=j9ay) code:j9ay |[GoogleDrive](https://drive.google.com/file/d/1lHOumI4wDsJfgGnDPOCXM6h_C1F5ZQEA/view?usp=sharing)|
 |:-----------:| :-----------: | :-----------: |:-----------: |
 |Night_Train|[Dropbox](https://www.dropbox.com/scl/fi/cw3ji53qxy18sepuk6wcp/NightRainDrop_Train.zip?rlkey=r2yn224ryek9wxkbchedeg13j&st=jzo93x80&dl=0)| [BaiduPan](https://pan.baidu.com/s/13x6-UzqxaJG7tKv2WMyMuQ?pwd=hmsw) code:hmsw| [GoogleDrive](https://drive.google.com/file/d/1_ruwsBCzbOEkpqcqHIeKoyCg6P2Sxclr/view?usp=sharing)|

## Validation Data (only for challenge) (do not distinguish day and night)
|Day + Night |[Dropbox](https://www.dropbox.com/scl/fo/74s4wpvhrx1ag0rvsuq9q/ALyqJEFapXZ7QBcOyV2uCSs?rlkey=hsu5ktfetpjh986rvkmxexule&st=k9m9wgxw&dl=0) |[BaiduPan](https://pan.baidu.com/s/1gwMC3uN6UrJ23xn24NNVdg?pwd=vali) code:vali | [GoogleDrive](https://drive.google.com/drive/folders/1KFInlF3cSZXxUIK4N9YvCC0B4N9lXAhB?usp=sharing)|
|:-----------:|:-----------:| :-----------: | :-----------: |

## Challenge Test Data (only for challenge) (do not distinguish day and night)
|Day + Night |[Dropbox](https://www.dropbox.com/scl/fo/5j5fb5pk7yn0tf61npq09/AMLYar9xGEfsD512emGBT80?rlkey=eusl5jkrhm5fdphzntmlzmj6r&st=mgw8u6ie&dl=0) |[BaiduPan](https://pan.baidu.com/s/1isQiEjfmnWCUb0EC0MTflQ?pwd=ntir) code:ntir | [GoogleDrive](https://drive.google.com/file/d/1yM2FwraaLx3Ql4CpvYqLYwM-DoONmC5e/view?usp=sharing)|
|:-----------:|:-----------:| :-----------: | :-----------: |

## RaindropClarity Test Data (for ECCV'2024) 
 |Day_Test|[Dropbox](https://www.dropbox.com/scl/fo/0nr4e0mt18xwhu97ukwnb/AJOJy0S8nrPHbojczZXDIR8?rlkey=uj45wkeixnnfum8qqshd9ugbx&st=fm8jouh1&dl=0)|[BaiduPan](https://pan.baidu.com/s/1np_4nM19-czzW3Pe3JjMsQ?pwd=dten) code:dten|[GoogleDrive](https://drive.google.com/drive/folders/1j2ldKp6jTFw2-tp3RlqMQvv2hcdJ4pYJ?usp=sharing)|
  |:-----------:| :-----------: | :-----------: |:-----------: |
 |Night_Test|[Dropbox](https://www.dropbox.com/scl/fo/bf0omjr89limztjkefc4e/AGBAedQMPeIbNr5In4F_6Sk?rlkey=4dtgospapvr42pujjvtczy6ul&st=mx5ny6l8&dl=0)|[BaiduPan](https://pan.baidu.com/s/110YZN6QfYLbiMYxr3gN6tQ?pwd=nten) code:nten|[GoogleDrive](https://drive.google.com/drive/folders/1lurHqpyZPG8xB96Gg5Qigcb3uYwG7Cwf?usp=sharing)|

## RaindropClarity Test Data with Ground Truth (for ECCV'2024) 

# Abstract @ ECCV’2024
 Existing raindrop removal datasets have two shortcomings. First, they consist of images captured by cameras with a focus on the background, leading to the presence of blurry raindrops. To our knowledge, none of these datasets include images where the focus is specifically on raindrops, which results in a blurry background. Second, these datasets predominantly consist of daytime images, thereby lacking nighttime raindrop scenarios. Consequently, algorithms trained on these datasets may struggle to perform effectively in raindrop-focused or nighttime scenarios. The absence of datasets specifically designed for raindrop-focused and nighttime raindrops constrains research in this area. In this paper, we introduce a large-scale, real-world raindrop removal dataset called Raindrop Clarity. Raindrop Clarity comprises 15,186 high-quality pairs/triplets (raindrops, blur, and background) of images with raindrops and the corresponding clear background images. There are 5,442 daytime raindrop images and 9,744 nighttime raindrop images. Specifically, the 5,442 daytime images include 3,606 raindrop- and 1,836 background-focused images. While the 9,744 nighttime images contain 4,838 raindrop- and 4,906 background-focused images. Our dataset will enable the community to explore background-focused and raindrop-focused images, including challenges unique to daytime and nighttime conditions. 

<p align="left">
  <img width=950" src="teaser/RaindropClarity.png">
</p>

## Pre-trained Models: [BaiduPan](https://pan.baidu.com/s/1tzJX--WD7YsYbpc9nGBQ0w?pwd=i3dg) code:i3dg and [Results](https://pan.baidu.com/s/1kVxJK0HgSDe5pglQ2uTj3A?pwd=outp) code:outp
| Model Name | Model Dropbox | Model BaiduPan | Results Dropbox | Results BaiduPan |
| :----: | :-----------: | :----------: |:---------------: |  :----------: |
| Raindrop + Restoration| [Dropbox](https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/ACnxDbcyX2K0trBdEJa4DdQ?rlkey=jn0wkbaf8d4xv8rqixhhuhymy&dl=0) | [BaiduPan](https://pan.baidu.com/s/1tzJX--WD7YsYbpc9nGBQ0w?pwd=i3dg) code:i3dg| [Dropbox](https://www.dropbox.com/scl/fo/mw63y7ss04404pb4vnd1r/AOanObkvsJmUFZEGYmud--c?rlkey=p9hjkawn7vn34hahycxjigv6l&st=aak7ifwh&dl=0) | [BaiduPan](https://pan.baidu.com/s/1kVxJK0HgSDe5pglQ2uTj3A?pwd=outp) code:outp|  

1. the checkpoint RainDrop_DiT_ddpm.pth.tar is patch_size=4, image_size:64,
[transformer2d.py](https://github.com/jinyeying/RaindropClarity/blob/b825ceeb6f2e1df6d33a5b57fe01dece43d3d98a/models/transformer2d.py#L455), follow patch_size=4

2. the checkpoint RainDrop_DiT2_ddpm.pth.tar is patch_size=2, image_size:64, 
[transformer2d.py](https://github.com/jinyeying/RaindropClarity/blob/b825ceeb6f2e1df6d33a5b57fe01dece43d3d98a/models/transformer2d.py#L452), follow patch_size=2

## Evaluation
```
python calculate_psnr_ssim_sid.py
```
please change `base_path`, `time_of_day`, `model_name` accordingly.

## Raindrop-focused or Background-focused?
The analysis code is available at [analyse/cal_rf_bf.py](https://github.com/jinyeying/RaindropClarity/blob/main/analyse/cal_rf_bf.py)
```
python cal_rf_bf.py
```
<p align="left">
  <img width=950" src="teaser/example.png">
</p>

## Test
```
bash run_eval_diffusion_day.sh
```
```
bash run_eval_diffusion_night.sh
```
Inside the script, please change `model_name` accordingly. 
```
CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_dit.py --sid "$sid"
```
```
CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_rdiffusion.py --sid "$sid"
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

## Acknowledgments
Code is implemented based [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion), we would like to thank them.

## License
The code and models in this repository are licensed under the MIT License for academic and other non-commercial uses.<br>
For commercial use of the code and models, separate commercial licensing is available. Please contact:
- Yeying Jin (jinyeying@u.nus.edu)
- Jonathan Tan (jonathan_tano@nus.edu.sg)

### Citation
If this work is useful for your research, please cite our paper. 
```BibTeX
@inproceedings{jin2024raindrop,
  title={Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal},
  author={Jin, Yeying and Li, Xin and Wang, Jiadong and Zhang, Yan and Zhang, Malu},
  booktitle={European Conference on Computer Vision},
  pages={1--17},
  year={2024},
  organization={Springer}
}
```
