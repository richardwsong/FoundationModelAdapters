import os
import glob
import numpy as np
import random
from torch.utils.data import Dataset
import cv2

EEG_INDEX_BASE_DIR = "data/SampleData/vigall_out/"
EEG_SPECTROGRAM_DIR = "data/SampleData/eeg_spectrograms/"
FMRI_SPECTROGRAM_DIR = "data/SampleData/fmri_spectrograms/"
EEG_INDEX_SOURCE_FORMAT = '*.npy'

FMRI_PROC_SCANS = ['sub01-scan01',
                   'sub01-scan02',
                   'sub02-scan01',
                   'sub02-scan02',
                   'sub03-scan01',
                   'sub03-scan02',
                   'sub04-scan01',
                   'sub04-scan02',
                   'sub05-scan01',
                   'sub06-scan01',
                   'sub07-scan01',
                   'sub08-scan01',
                   'sub08-scan02',
                   'sub09-scan01',
                   'sub10-scan01',
                   'sub11-scan01']

FMRI_PROC_SCANS_TRAIN = ['sub01-scan01',
                        'sub01-scan02',
                        'sub02-scan01',
                        'sub02-scan02',
                        'sub03-scan01',
                        'sub03-scan02',
                        'sub04-scan01',
                        'sub04-scan02',
                        'sub05-scan01',
                        'sub06-scan01',
                        'sub09-scan01',
                        'sub10-scan01',
                        'sub11-scan01']

FMRI_PROC_SCANS_TEST = ['sub07-scan01',
                        'sub08-scan01',
                        'sub08-scan02']

SCAN_TASK_DICT = {'ect': 'ectp'}

class SampleDatasetConfig(object):
    def __init__(self):
        self.default_value = 1
        self.num_classes = 2
        self.name = "SampleData"

class SampleDataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
    ):
        assert split_set in ["train", "test", "zero_shot"]
        self.dataset_config = dataset_config
        self.eeg_index_base_dir = EEG_INDEX_BASE_DIR
        self.scan_task_dict = SCAN_TASK_DICT
        self.eeg_spectrogram_dir = EEG_SPECTROGRAM_DIR
        self.fmri_spectrogram_dir = FMRI_SPECTROGRAM_DIR
        self.eeg_index_source_format = EEG_INDEX_SOURCE_FORMAT
        
        all_scan_names = FMRI_PROC_SCANS
        train_scan_names = FMRI_PROC_SCANS_TRAIN
        test_scan_names = FMRI_PROC_SCANS_TEST

        all_subject_names = {}
        for scan_name in all_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in all_subject_names:
                all_subject_names[subject_name] = []
            all_subject_names[subject_name].append(scan_name)

        train_subject_names = {}
        for scan_name in train_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in train_subject_names:
                train_subject_names[subject_name] = []
            train_subject_names[subject_name].append(scan_name)
        
        test_subject_names = {}
        for scan_name in test_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in test_subject_names:
                test_subject_names[subject_name] = []
            test_subject_names[subject_name].append(scan_name)
       
        np.random.seed(0)
        random.seed(0)

        if split_set == "train":
            self.scan_names = train_scan_names
            random.shuffle(self.scan_names)
        elif split_set == "test":
            self.scan_names = test_scan_names
            random.shuffle(self.scan_names)
        elif split_set == "zero_shot":
            self.scan_names = all_scan_names
            random.shuffle(self.scan_names)

        result_scan_names = {}
        for scan_name in self.scan_names:
            subject_name = scan_name[:8]
            if subject_name in result_scan_names:
                result_scan_names[subject_name] += 1
            else:
                result_scan_names[subject_name] = 1

    def __len__(self):
        return len(self.scan_names)
    
    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        eeg_index_path = glob.glob(os.path.join(self.eeg_index_base_dir, scan_name+self.eeg_index_source_format))
        eeg_index = np.load(eeg_index_path[0])

        eeg_spectrogram_path = glob.glob(os.path.join(self.eeg_spectrogram_dir, scan_name+"*.png"))[0]
        eeg_spectogram_img = cv2.imread(eeg_spectrogram_path)

        # Read multiple fmri spectrograms from different ROIs
        fmri_spectrogram_paths = glob.glob(os.path.join(self.fmri_spectrogram_dir, scan_name+"*.png"))
        
        # Get at least 2 fMRI spectrograms if available, otherwise duplicate the first one
        num_rois = 2  # We can adjust this parameter later
        fmri_spectrograms_list = []
        
        for i in range(min(num_rois, len(fmri_spectrogram_paths))):
            fmri_spectrograms_list.append(cv2.imread(fmri_spectrogram_paths[i]))
            
        # If we don't have enough spectrograms, duplicate the first one
        while len(fmri_spectrograms_list) < num_rois:
            fmri_spectrograms_list.append(cv2.imread(fmri_spectrogram_paths[0]))
            
        # Stack the spectrograms along a new dimension
        if num_rois > 1:
            fmri_spectrograms = np.stack(fmri_spectrograms_list, axis=0)
            print(f"fmri_spectrograms shape: {fmri_spectrograms.shape}")
        else:
            fmri_spectrograms = np.expand_dims(fmri_spectrograms_list[0], axis=0)
            print(f"fmri_spectrograms shape: {fmri_spectrograms.shape}")

        ret_dict = {}
        ret_dict["eeg_index"] = eeg_index
        ret_dict["eeg_spectrogram_img"] = np.array(eeg_spectogram_img)
        ret_dict["fmri_spectrogram_imgs"] = fmri_spectrograms
        return ret_dict