# 2025BHVU: Foundation Model Adapters for Spectrogram Latent Features Extraction

## Short title: Foundation Model Adapters

## Leaders and Collaborators
Chang Li ([homepage](https://alexandrachangli.github.io/))

Collaborators:
1. Yamin Li ([homepage](https://soupeeli.github.io/))

## Project Summary
We leverage knowledge from pre-trained vision foundation models and develop adapters to extract brain patterns’ intrinsic features embedded in spectrograms.

## Project Description
*Topic:* Leveraging foundation models’ pre-trained knowledge to conduct spectrogram analysis of EEG and fMRI resting state data of healthy subjects.

*Objective:* Differentiate drowsy and alertness states based on EEG and fMRI spectrogram features.

*Potential Contribution:* Adapt foundation models’ robust, pre-trained knowledge of the brain-imaging domain can potentially generate high-quality latent space with good discriminability among subjects and brain states.

*Get Started:* Download the sample code and data. Follow the tutorial in this readme file to set up environments and run the basic pipeline. Integrate foundation models for feature extraction. Visualization and analysis.

*Key Resources:* Background theory: foundation models’ official tutorials, vision detection models. Implementation: PyTorch documentations. ChatGPT is your good friend if you use it wisely!

## Goals for the BrainHack Global
1. Get a basic understanding of brain imaging spectrograms (not computationally intensive).
    a. Understand how brain imaging spectrograms are generated.
    b. Understand the differences between EEG and fMRI spectrograms, as well as the pros and cons of these two modalities.
    c. Differentiate drowsy and alert states from EEG and fMRI spectrograms manually, and think about what might contribute to the differences in latent space.
2. Compare different foundation models’ abilities in capturing brain images’ intrinsic information and the quality of latent space (GPU preferred; CPU with a large memory is also fine!).
    a. Experiment with the CLIP feature extractor in the sample code.
    b. Integrate visual foundation models with open-source checkpoints/APIs (e.g. DALL-E, SAM) as feature extractors.
    c. Visualize the features with UMAP (sample code provided) or other methods.
3. Implement fine-tuning blocks for downstream tasks (e.g. classification) based on the latent space.
    a. Experiment with the adapter for CLIP features in the sample code. Experiment with the MLP predictor for alertness/drowsiness classification.
    b. Modify the adapter layers to enhance discriminability between features from different brain states. Modify the MLP predictor to enhance model performance.
    c. Experiment with features from other foundation models.


## Good First Issues
1. Set up environments and run the model pipeline.
2. How to extract visual features from spectrogram images.
3. How to enhance discriminability between different brain states in the latent feature space.


## Communication Channels
BrainHack Vanderbilt Discord channel.

## Skills
1. Able to set up Python deep learning environment (IDE, Anaconda). Have Python programming experience or at least intermediate knowledge level of other languages.
2. A beginner-level understanding of visual/language foundation models, e.g. CLIP, DALL-E, SAM, no prior experience required.
3. Intermediate deep learning experience preferred.
4. Beginner-level Git knowledge required. Intermediate Git experience preferred.


## What Will Participants Learn
1. Integration of foundation models into deep learning pipeline.
2. Basic EEG/fMRI spectrogram interpretation.
3. Understanding of EEG/fMRI latent space.


## Data
Paired EEG and fMRI spectrograms are generated from part of the internal dataset of [Neurdylab](https://www.cchanglab.net/home). 
The generated spectrograms and ground truth is released at [OSF](https://osf.io/hxaz4/) with DOI: Identifier: DOI 10.17605/OSF.IO/HXAZ4.

**Base folder:** *data/SampleData/*

**EEG spectrograms:** path: *data/SampleData/eeg_spectrograms*. Every EEG scan has a corresponding spectrogram.

**fMRI spectrograms:** path: *data/SampleData/fmri_spectrograms*. Every fMRI scan has spectrograms regarding to 64 ROIs from the 

**Brain vigilance state ground truth:** path: *data/SampleData/vigall_out*. 
Binary groundtruth is generated in a sliding-window manner with window size 30, step size 15. 1: vigilance; 0: drowsy.


## Number of Collaborators and Credits
3 Collaborators. Collaborators contributing to method design/experiments/visualizations/writings will become coauthors in follow-up publications.

## Metadata
1. Computational Resources: The codebase is implemented in both the CPU and GPU versions. Change the arguments in the running scripts for switching computational devices. Sample data size: < 61M. Memory usage in training: 3000MiB.
2. Code Implementation: The pipeline is implemented following [1][2][3].
3. Data Preparation: EEG spectrograms are generated following the work of [4], and fMRI spectrograms are generated following the work of [5]. fMRI ROIs are extracted based on the Dictionaries of Functional
Modes atlas [6]. Brain states ground truth is computed via the VIGALL algorithm [7]. Ground truth label in dataloader: eeg_index. 


## Environment Configs, Training and Testing the Model, Visualization
**Step 0:** Download the data and change the data loading path. Open a terminal and enter the working folder. 

**Step 1:** Run the conda environment setting command: 
```bash
bash enviro_configs/enviro_config.sh 
conda activate brain_clip_adapter
```
**Step 2:** Training. Change main.py for the GPU index and the argument '--train_device' for CPU/GPU selection.
```bash
bash scripts/brain_clip_adapter_000_train.sh
```
Argument explanation:
```bash
#!/bin/bash 
# Copyright (c) Facebook, Inc. and its affiliates.
python main.py \
--model_name brain_clip_adapter \ # your developed model
--dataset_name SampleDataset \ 
--train_device gpu \
--max_epoch 5 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.1 \ # loss weight, feel free to add additional loss here. Add argument --l2_loss for l2 regularization.
--checkpoint_dir outputs/brain_clip_adapter_000 \
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 2 \
--ngpus 1 \
# > logs/brain_clip_adapter_000_train.txt \ 
```

**Step 3:** Testing and Visualization

For testing, run:
```bash
bash scripts/brain_clip_adapter_000_test.sh
``` 
Argument explanation:
```bash
#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
python main.py \
--model_name brain_clip_adapter \
--dataset_name SampleDataset \
--test_only \ # use --zero_shot for testing the whole dataset
--test_ckpt outputs/brain_clip_adapter_000/checkpoint_best.pth \
--test_visualize \ # umap visualization of the latent space
--test_logger outputs/brain_clip_adapter_000/SampleDataset \ 
# > logs/brain_clip_adapter_000_SampleDataset_zeroshot.txt \
```
To check the visualizations, run the following:
```bash
tensorboard --logdir=outputs/brain_clip_adapter_000
```
**Step 4:** You are free to modify the model for design (e.g. adapter layers, foundation models used, inference methods), and the criterion.py for training loss functions for your experiment.

**Step 5:** Compare what we have done with foundation models to naive CNN layers, what is the difference in the latent space? How can we increase the discriminability?

## References
[1] Misra I, Girdhar R, Joulin A. An end-to-end transformer model for 3d object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 2906-2917.

[2] Lu Y, Xu C, Wei X, et al. Open-vocabulary point-cloud object detection without 3d annotation[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 1190-1199.

[3] Gao P, Geng S, Zhang R, et al. Clip-adapter: Better vision-language models with feature adapters[J]. International Journal of Computer Vision, 2024, 132(2): 581-595.

[4] Prerau M J, Brown R E, Bianchi M T, et al. Sleep neurophysiological dynamics through the lens of multitaper spectral analysis[J]. Physiology, 2017, 32(1): 60-92.

[5] Song C, Boly M, Tagliazucchi E, et al. fMRI spectral signatures of sleep[J]. Proceedings of the National Academy of Sciences, 2022, 119(30): e2016732119.

[6] Dadi K, Varoquaux G, Machlouzarides-Shalit A, et al. Fine-grain atlases of functional modes for fMRI analysis[J]. NeuroImage, 2020, 221: 117126.

[7] Olbrich S, Fischer M M, Sander C, et al. Objective markers for sleep propensity: comparison between the Multiple Sleep Latency Test and the Vigilance Algorithm Leipzig[J]. Journal of sleep research, 2015, 24(4): 450-457.

<!-- GB/T 7714 -->