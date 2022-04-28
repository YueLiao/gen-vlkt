#!/usr/bin/env bash

set -x
EXP_DIR=exps/vcoco_gen_vlkt_l_r101_dec_6layers

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r101-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet101 \
        --num_queries 64 \
        --dec_layers 6 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --mimic_loss_coef 20
