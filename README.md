# RaindropClarity (ECCV'2024)

## Introduction
> [Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](https://arxiv.org/abs/2407.16957)<br>
> European Conference on Computer Vision (ECCV'2024)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.16957)

[[Poster]](https://github.com/jinyeying/RaindropClarity/blob/main/poster_slides/RaindropClarity_poster.pdf) 
[[Slides]](https://github.com/jinyeying/RaindropClarity/blob/main/poster_slides/RaindropClarity_PPT.pdf) 
[[Video]](https://www.youtube.com/watch?v=LSGvCuT46XU)

### Abstract
 Existing raindrop removal datasets have two shortcomings. First, they consist of images captured by cameras with a focus on the background, leading to the presence of blurry raindrops. To our knowledge, none of these datasets include images where the focus is specifically on raindrops, which results in a blurry background. Second, these datasets predominantly consist of daytime images, thereby lacking nighttime raindrop scenarios. Consequently, algorithms trained on these datasets may struggle to perform effectively in raindrop-focused or nighttime scenarios. The absence of datasets specifically designed for raindrop-focused and nighttime raindrops constrains research in this area. In this paper, we introduce a large-scale, real-world raindrop removal dataset called Raindrop Clarity. Raindrop Clarity comprises 15,186 high-quality pairs/triplets (raindrops, blur, and background) of images with raindrops and the corresponding clear background images. There are 5,442 daytime raindrop images and 9,744 nighttime raindrop images. Specifically, the 5,442 daytime images include 3,606 raindrop- and 1,836 background-focused images. While the 9,744 nighttime images contain 4,838 raindrop- and 4,906 background-focused images. Our dataset will enable the community to explore background-focused and raindrop-focused images, including challenges unique to daytime and nighttime conditions. 

## RaindropClarity Dataset
 |Day_Train   |[Dropbox](https://www.dropbox.com/scl/fi/qes7r934c10qzb21funoj/DayRainDrop_Train.zip?rlkey=bdqa53wgvmhj9x1yf40q0c1p7&st=4taffjkx&dl=0) | [BaiduPan](https://pan.baidu.com/s/1-vwhYA7jEDPAYHlznhcHCA?pwd=j9ay) code:j9ay |[GoogleDrive](https://drive.google.com/file/d/1lHOumI4wDsJfgGnDPOCXM6h_C1F5ZQEA/view?usp=sharing)|
 |:-----------:| :-----------: | :-----------: |:-----------: |
 |Day_Test|[Dropbox](https://www.dropbox.com/scl/fi/ft0pvf8pdkueedxp9068n/DayRainDrop_Test_woGT.zip?rlkey=jneh1i9no2iedyttv6514jcen&st=jnsnyslw&dl=0)|[BaiduPan](https://pan.baidu.com/s/1np_4nM19-czzW3Pe3JjMsQ?pwd=dten) code:dten|[GoogleDrive](https://drive.google.com/file/d/1v-0oJYxyaRnaOLbrcydZF5juI8XMQthc/view?usp=sharing)|
 |Night_Train|[Dropbox](https://www.dropbox.com/scl/fi/cw3ji53qxy18sepuk6wcp/NightRainDrop_Train.zip?rlkey=r2yn224ryek9wxkbchedeg13j&st=jzo93x80&dl=0)| [BaiduPan](https://pan.baidu.com/s/13x6-UzqxaJG7tKv2WMyMuQ?pwd=hmsw) code:hmsw| [GoogleDrive](https://drive.google.com/file/d/1_ruwsBCzbOEkpqcqHIeKoyCg6P2Sxclr/view?usp=sharing)|
 |Night_Test|[Dropbox](https://www.dropbox.com/scl/fi/9y496nqvcuqh8sis24h1e/NightRainDrop_Test_woGT.zip?rlkey=6jcid5v1gjtk0u9o1gcrrliuu&st=4k95zm81&dl=0)|[BaiduPan](https://pan.baidu.com/s/110YZN6QfYLbiMYxr3gN6tQ?pwd=nten) code:nten|[GoogleDrive](https://drive.google.com/file/d/1TaYnMhO79LmGoGDbmaeaCloLlIP4geKs/view?usp=sharing)|


## Pre-trained Models: [BaiduPan](https://pan.baidu.com/s/1tzJX--WD7YsYbpc9nGBQ0w?pwd=i3dg) code:i3dg and [Results](https://pan.baidu.com/s/1kVxJK0HgSDe5pglQ2uTj3A?pwd=outp) code:outp
| Model Name | Model Dropbox | Model BaiduPan | Results Dropbox | Results BaiduPan |
| :----: | :-----------: | :----------: |:---------------: |  :----------: |
| Raindrop + Restoration| [Dropbox](https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/ACnxDbcyX2K0trBdEJa4DdQ?rlkey=jn0wkbaf8d4xv8rqixhhuhymy&dl=0) | [BaiduPan](https://pan.baidu.com/s/1tzJX--WD7YsYbpc9nGBQ0w?pwd=i3dg) code:i3dg| [Dropbox]() | [BaiduPan](https://pan.baidu.com/s/1kVxJK0HgSDe5pglQ2uTj3A?pwd=outp) code:outp|  

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
@article{jin2024raindrop,
  title={Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal},
  author={Jin, Yeying and Li, Xin and Wang, Jiadong and Zhang, Yan and Zhang, Malu},
  journal={arXiv preprint arXiv:2407.16957},
  year={2024}
}
```
