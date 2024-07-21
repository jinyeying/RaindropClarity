#!/bin/bash

sids=(
    '00060' '00129' '00144' '00147' '00151' '00161' '00164' '00168'
    '00173' '00174' '00176' '00181' '00184' '00185' '00186' '00187' '00188' '00189'
    '00190' '00191' '00192' '00193' '00194' '00195' '00196' '00197'
)

for sid in "${sids[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python eval_diffusion_night_restomer.py --sid "$sid"
done
# CUDA_VISIBLE_DEVICES=4 python eval_diffusion_night_dit.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=5 python eval_diffusion_night_rdiffusion.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion_night_restomer.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion_night_uformer.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion_night_onego.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=3 python eval_diffusion_night_idt.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=3 python eval_diffusion_night_icra.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion_night_atgan.py --sid "$sid"
