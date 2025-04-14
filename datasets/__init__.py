from .SampleDataset import SampleDataset, SampleDatasetConfig

DATASET_FUNCTIONS = {
    "SampleDataset": [SampleDataset, SampleDatasetConfig],
}

def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
            fmri_index=args.dataset_fmri_index,
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="test",
            fmri_index=args.dataset_fmri_index,
        ),
        "zero_shot": dataset_builder(
            dataset_config,
            split_set="zero_shot",
            fmri_index=args.dataset_fmri_index,
        ),
    }
    return dataset_dict, dataset_config


def build_dataset_zero_shot(args):
    train_dataset_builder = DATASET_FUNCTIONS[args.zero_shot_train][0]
    train_dataset_config = DATASET_FUNCTIONS[args.zero_shot_train][1]()
    test_dataset_builder = DATASET_FUNCTIONS[args.zero_shot_test][0]
    test_dataset_config = DATASET_FUNCTIONS[args.zero_shot_test][1]()
    train_dataset_dict = {
        "train": train_dataset_builder(
            train_dataset_config,
            split_set="train",
            fmri_index=args.dataset_fmri_index,
        ),
        "test": train_dataset_builder(
            train_dataset_config,
            split_set="test",
            fmri_index=args.dataset_fmri_index,
        ),
        "zero_shot": train_dataset_builder(
            train_dataset_config,
            split_set="zero_shot",
            fmri_index=args.dataset_fmri_index,
        ),
    }
    test_dataset_builder = {
        "train": test_dataset_builder(
            test_dataset_config,
            split_set="train",
            fmri_index=args.dataset_fmri_index,
        ),
        "test": test_dataset_builder(
            test_dataset_config,
            split_set="test",
            fmri_index=args.dataset_fmri_index,
        ),
        "zero_shot": test_dataset_builder(
            test_dataset_config,
            split_set="zero_shot",
            fmri_index=args.dataset_fmri_index,
        ),
    }
    return train_dataset_dict, train_dataset_config, test_dataset_builder, test_dataset_config