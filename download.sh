#!/bin/bash

## download weights of the modified decoder:
mkdir -p ./weights/LaWa && cd weights/LaWa
wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lawa/pretrained-checkpoints/last.ckpt
## download weights of the original decoder:
cd ..
mkdir -p first_stage_models && cd first_stage_models
wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lawa/pretrained-checkpoints/first_stage_KL-f8.ckpt
cd ../..
mkdir -p ./data && cd data
wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lawa/data/dataset.zip
unzip dataset.zip
