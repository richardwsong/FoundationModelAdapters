#!/bin/bash

for FMRI_INDEX in {0..12}; do
  # Pad the index with leading zeros (e.g., 4 → 004)
  PADDED_INDEX=$(printf "%03d" $FMRI_INDEX)
  echo "Running for fMRI index: ${PADDED_INDEX}"

  OUTPUT_DIR="outputs/brain_clip_adapter_${PADDED_INDEX}"

  # Training
  python main.py \
  --model_name brain_clip_adapter \
  --dataset_name SampleDataset \
  --dataset_fmri_index ${FMRI_INDEX} \
  --train_device cpu \
  --max_epoch 5 \
  --base_lr 7e-4 \
  --loss_fmri_prediction_weight 0.1 \
  --checkpoint_dir ${OUTPUT_DIR} \
  --dataset_num_workers 1 \
  --log_metrics_every 5 \
  --batchsize_per_gpu 2 \
  --ngpus 1

  # Testing
  python main.py \
  --model_name brain_clip_adapter \
  --dataset_name SampleDataset \
  --dataset_fmri_index ${FMRI_INDEX} \
  --test_only \
  --test_ckpt ${OUTPUT_DIR}/checkpoint_best.pth \
  --test_visualize \
  --test_logger ${OUTPUT_DIR}/SampleDataset

  echo "Finished index ${PADDED_INDEX}"
  echo "----------------------------------------"
done