#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_gen_vlkt_m_r101_dec_3layers

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r101-pre-2branch-hico.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet101 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --mimic_loss_coef 20
