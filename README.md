[**Deformable-DETR**](http://arxiv.org/abs/2010.04159): Deformable Transformers for End-to-End Object Detection


# Deformable-DETR

This an implementation of Deformable-DETR. Codes are based on [DETR](https://github.com/facebookresearch/detr) project.
My code is inspired by his/her [work]( https://github.com/Windaway/Deformable-Attention-for-Deformable-DETR/blob/main/DFMAtt.py). Many thanks.

# Preparation

For DETR stuffs, etc. data preparation, evaluation, and others , please refer to 
[DETR](https://github.com/facebookresearch/detr).

# Training

My machine is equipped with two GTX 2080TIs. Below is the training script for DDP training.
```shell script
bash train.sh
```

For single gpu training, try the code below

```python
python main.py
--output_dir my_output \
--coco_path ~/dev/data/coco \
--lr 0.0002 \
--lr_backbone 0.00001 \
--num_queries 300 \
--batch_size 1 \
--enc_layers 6 \
--dec_layers 6 \
--no_aux_loss \
--amp
```

If you do not need AMP, just remove this flag.

# Change logs
- 2020-11-30
  - add focal loss for classification

- 2020-11-29
  - integrate MS-Deformable-Attention into DETR architecture
  - modify transfomer's implementation to be adapted to Deformable-Attention
  - add image mask to MS-Deformable-Attention
  - add automatic mixed precision training
  - use adam for the optimizer
  - change lr for projection layers

- 2020-11-24
  - add scale embedding
  - change remove outer loop for scales
  - add backbone modifications for returning multi-scale feature maps
  - add test code for using Deformable-Attention module

- 2020-11-22 
  
  - add Multi-scale Deformabe Attention Module