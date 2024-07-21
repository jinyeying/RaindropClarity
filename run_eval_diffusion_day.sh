#!/bin/bash

sids=(
    '00007' '00008' '00011' '00015' '00023' '00032' '00041' '00050'
    '00055' '00064' '00070' '00074' '00078' '00081' '00083' '00096' '00099'
    '00101' '00121' '00146' '00150' '00156' '00157' '00161' '00169' '00172' '00192' '00199'
)

for sid in "${sids[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_restomer.py --sid "$sid"
done
# CUDA_VISIBLE_DEVICES=7 python eval_diffusion_day_dit.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=6 python eval_diffusion_day_rdiffusion.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_restomer.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_uformer.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_onego.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_idt.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion_day_icra.py --sid "$sid"
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion_day_atgan.py --sid "$sid"