#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --node_rank 0  --use_env main.py --amp --output_dir my_output --coco_path ~/dev/data/coco