python main.py \
--model_name brain_cnn_adapter \
--dataset_name SampleDataset \
--train_device cpu \
--max_epoch 5 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.1 \
--checkpoint_dir outputs/brain_cnn_adapter_000 \
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 2 \
--ngpus 1 \
# > logs/brain_cnn_adapter_000_train.txt \