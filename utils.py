import torch
import numpy as np

from sklearn.utils import shuffle

SEQU_SIZE = 200 # will be resize to 200

def load_dataset(path):
    dataset = []
    labels = []
    with open(path, "r") as f:
        for index, line in enumerate(f):
            dataset.append(int(line.split(",")[0]))
            labels.append(float(line.split(",")[1]))
            if index >= 4_000_000: # will be resize to 40M
                break

    return dataset, labels

def create_seq(dataset, labels, seq_size):
    dataset_result = []
    labels_result = []
    data_batch = []
    label_batch = []

    for index in range(len(dataset)):
        if len(data_batch) == SEQU_SIZE:
            dataset_result.append(data_batch)
            # dataset_result.append(torch.Tensor(data_batch))
            labels_result.append(label_batch[-1])
            data_batch = []
            label_batch = []
        else:
            data_batch.append(dataset[index])
            label_batch.append(labels[index])

    dataset_result, labels_result = shuffle(dataset_result, labels_result)
    return torch.tensor(dataset_result, dtype=torch.int16).unsqueeze(1), torch.tensor(labels_result, dtype=torch.float32)
    # return dataset_result, torch.tensor(labels_result, dtype=torch.int64)

def split(dataset, labels, test_ratio=0.2):
    X_train = dataset[:int(len(dataset) * (1-test_ratio))]
    y_train = labels[:int(len(dataset) * (1-test_ratio))]

    X_test = dataset[int(len(dataset) * (1-test_ratio)):]
    y_test = labels[int(len(dataset) * (1-test_ratio)):]
    return X_train, y_train, X_test, y_test