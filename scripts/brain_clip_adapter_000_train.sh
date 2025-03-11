#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name brain_clip_adapter \
--dataset_name SampleDataset \
--train_device cpu \
--max_epoch 5 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.1 \
--checkpoint_dir outputs/brain_clip_adapter_002 \
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 1 \
--ngpus 1 \
# > logs/brain_clip_adapter_000_train.txt \