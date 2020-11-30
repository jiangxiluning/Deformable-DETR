#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --node_rank 0  --use_env main.py \
--output_dir my_output \
--coco_path ~/dev/data/coco \
--lr 0.0002 \
--lr_backbone 0.00001 \
--num_queries 100 \
--batch_size 1 \
--enc_layers 6 \
--dec_layers 6 \
--no_aux_loss \
--amp
