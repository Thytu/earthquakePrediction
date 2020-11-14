import torch
import numpy as np

SEQU_SIZE = 100

def load_dataset(path):
    dataset = []
    labels = []
    with open(path, "r") as f:
        for index, line in enumerate(f):
            dataset.append(line.split(",")[0])
            labels.append(line.split(",")[1])
            if index >= 20_000_000:
                break

    return dataset, labels

def create_seq(dataset, labels, seq_size):
    print(np.array(dataset).shape, np.array(labels).shape)
    dataset_result = []
    labels_result = []
    data_batch = []
    label_batch = []

    for index in range(len(dataset)):
        if len(data_batch) == SEQU_SIZE:
            dataset_result.append(data_batch)
            labels_result.append(label_batch)
            data_batch = []
            label_batch = []
        else:
            data_batch.append(dataset[index])
            label_batch.append(labels[index])

    print(np.array(dataset_result).shape, np.array(labels_result).shape)
    return torch.Tensor(dataset_result), torch.tensor(labels_result, dtype=torch.int64)