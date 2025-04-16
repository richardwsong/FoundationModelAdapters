# check_class_balance.py
import argparse
from collections import Counter
import os
import sys

# Fix the path so it can find sampledataset.py
sys.path.append(os.path.join(os.path.dirname(__file__), "datasets"))

from datasets.SampleDataset import SampleDataset, SampleDatasetConfig

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")  # add option for train/test
args = parser.parse_args()

config = SampleDatasetConfig()
dataset = SampleDataset(dataset_config=config, split_set=args.split)

# You’ll need to define how to get the label; assuming it's stored in eeg_index[0][-1]
labels = []
for item in dataset:
    # TODO: Replace with your actual label logic
    # For example, if it's binary classification and label is the last EEG index:
    label = int(item["eeg_index"][-1][0])
    labels.append(label)

print("Class distribution:", Counter(labels))
