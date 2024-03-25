import torch
import torch.nn as nn
from torch.utils.data import Dataset

# define the dataset class
class GeoDataset(Dataset):
    def __init__(self, grouped_data):
        # initialize the dataset with grouped data
        self.grouped_data = grouped_data

    def __len__(self):
        # return the length of the dataset
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # retrieve a group based on the index
        group = self.grouped_data[idx]
        # this code means that the group is shuffled before being returned
        # group = group.sample(frac=1).reset_index(drop=True)  # close it for the OLST
        # Extract features and labels from the shuffled group
        features = torch.tensor(group.iloc[:, 1:-1].values, dtype=torch.float)
        labels = torch.tensor(group.iloc[:, -1].values, dtype=torch.long)
        seq_length = len(features)
        # return the features, labels, and sequence length
        return features, labels, seq_length


# define a custom collate function for the dataloader, custom means the sequence length is different
def custom_collate_fn(batch):
    # extract features, labels, and sequence lengths from the batch
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    seq_lengths = torch.tensor([item[2] for item in batch])
    # pad the features and labels to the maximum sequence length in the batch
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    # create a range tensor that matches the length of the sequences
    max_length = padded_features.size(1)
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(padded_features.size(0), -1)
    # create a mask where true values represent valid data points
    mask = range_tensor < seq_lengths.unsqueeze(1)
    # return the padded features, labels, and the mask
    return padded_features, padded_labels, mask