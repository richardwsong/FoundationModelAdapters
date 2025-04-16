#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name brain_clip_adapter \
--dataset_name SampleDataset \
--dataset_fmri_index 3 \
--test_only \
--test_ckpt outputs/brain_clip_adapter_004/checkpoint_best.pth \
--test_visualize \
--test_logger outputs/brain_clip_adapter_004/SampleDataset \
# > logs/brain_clip_adapter_000_SampleDataset_zeroshot.txt \