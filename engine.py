# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import logging
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from umap import UMAP
from torch.distributed.distributed_c10d import reduce
from utils.ac_re_calculator import ACRECalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):

    acre_calculator = ACRECalculator(
        dataset_config=dataset_config,
    )

    curr_iter = curr_epoch * len(dataset_loader)

    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        optimizer.zero_grad()
        inputs = {
            "eeg_index": batch_data_label["eeg_index"].float(),
            "eeg_spectrogram_img": batch_data_label["eeg_spectrogram_img"].float(),
            "fmri_spectrogram_imgs": batch_data_label["fmri_spectrogram_imgs"].float(),
        }
        outputs = model(inputs)

        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        loss_reduced = all_reduce_average(loss)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss = loss.float()
        
        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)
        
        loss.requires_grad_(True)

        if args.l2_loss:
            lambda_l2 = 1e-4 
            l2_loss = lambda_l2 * sum(p.pow(2.0).sum() for p in model.parameters())
            l2_loss.requires_grad_(True)
            total_loss = loss + l2_loss
            total_loss.backward()
        else:
            loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            acre_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()
        
    return acre_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):
    print("Starting evaluation...")
    
    acre_calculator = ACRECalculator(
        dataset_config=dataset_config,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        print(f"Evaluation batch {batch_idx}")
        print(f"eeg_index shape: {batch_data_label['eeg_index'].shape}")
        print(f"eeg_index values (first few): {batch_data_label['eeg_index'][:2]}")

        # Check vigilance labels
        vigilance_labels = batch_data_label.get('vigilance_label')
        if vigilance_labels is not None:
            print(f"vigilance_label distribution: {torch.bincount(vigilance_labels)}")

        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "eeg_index": batch_data_label["eeg_index"].float(),
            "eeg_spectrogram_img": batch_data_label["eeg_spectrogram_img"].float(),
            "fmri_spectrogram_imgs": batch_data_label["fmri_spectrogram_imgs"].float(),
        }
        outputs = model(inputs, is_train=False)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers GT tensor across all ranks
        outputs = all_gather_dict(outputs["predictions"])
        batch_data_label = all_gather_dict(batch_data_label)
        acre_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
        
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

        fmri_feats_total = []
        eeg_index_total = []

        if args.test_visualize:
            for idx, data in enumerate(dataset_loader.dataset):
                inputs = {
                    "eeg_index": torch.tensor(np.expand_dims(data["eeg_index"], axis=0)).float(),
                    "eeg_spectrogram_img": torch.tensor(np.expand_dims(data["eeg_spectrogram_img"], axis=0)).float(),
                    "fmri_spectrogram_imgs": torch.tensor(np.expand_dims(data["fmri_spectrogram_imgs"], axis=0)).float(),
                }
                outputs = model(inputs, is_train=False)
                fmri_feats_total.append(np.array(outputs["fmri_feats"]))
                eeg_index_total.append(np.array(data["eeg_index"]))

            fmri_feats_total = np.concatenate(fmri_feats_total, axis=0)
            eeg_index_total = np.concatenate(eeg_index_total, axis=0)

            umap_2d_fmri_feats = UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='cosine', init='random', random_state=0)
            proj_fmri_feats_2d = umap_2d_fmri_feats.fit_transform(fmri_feats_total)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(proj_fmri_feats_2d[:, 0], proj_fmri_feats_2d[:, 1], c=eeg_index_total, cmap='viridis', s=2)
            plt.colorbar(scatter, label='Label')
            plt.title("UMAP Visualization of fmri_feats")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            
            # Show the plot immediately
            print("\nDisplaying UMAP visualization...\n")
            plt.show()
            
            # Continue with the TensorBoard logging
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            comparison_img = Image.open(buf)
            comparison_img = np.array(comparison_img)
            comparison_img = torch.tensor(comparison_img).permute(2, 0, 1)
            logger.log_image(comparison_img, prefix='UMAP 2d visualization of fmri_feats')
            plt.close()
    return acre_calculator
