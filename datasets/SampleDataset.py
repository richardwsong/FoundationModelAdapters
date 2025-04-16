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
        eeg_index_path = glob.glob(os.path.join(self.eeg_index_base_dir, scan_name + self.eeg_index_source_format))
        eeg_index = np.load(eeg_index_path[0])  # shape: (45, 1) or similar

        eeg_spectrogram_path = glob.glob(os.path.join(self.eeg_spectrogram_dir, scan_name + "*.png"))[0]
        eeg_spectrogram_img = cv2.imread(eeg_spectrogram_path)

        fmri_spectrogram_path = glob.glob(os.path.join(self.fmri_spectrogram_dir, scan_name + "*.png"))
        fmri_spectrograms = cv2.imread(fmri_spectrogram_path[0])

        # Add this line — compute a single label (e.g., binary: 0 or 1)
        vigilance_label = int(np.median(eeg_index))

        return {
            "eeg_index": eeg_index,
            "eeg_spectrogram_img": np.array(eeg_spectrogram_img),
            "fmri_spectrogram_imgs": np.array(fmri_spectrograms),
            "vigilance_label": vigilance_label,
        }
