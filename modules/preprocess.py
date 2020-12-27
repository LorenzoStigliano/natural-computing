from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch

def load_tensors(dataloc, shuffle=True, val_prop=0.3, test_prop=0.2, phi=lambda x: x, device="cpu"):
    """
    Reads torch tensors from twospirals.dat and splits them into the appropriate datasets,
    whilst also pushing these variables to the GPU if desired.
    """
    data = np.loadtxt(dataloc)

    dataset_size = len(data)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    assert val_prop + test_prop >= 0 and val_prop + test_prop <= 0.5

    train_prop = 1 - val_prop - test_prop

    train_split = int(np.floor(train_prop * dataset_size))
    val_split = int(np.floor((train_prop+val_prop) * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_data = torch.from_numpy(data[train_indices]).float()
    val_data = torch.from_numpy(data[val_indices]).float()
    test_data = torch.from_numpy(data[test_indices]).float()

    x_train = phi(train_data[:, :2]).to(device)
    y_train = train_data[:, 2:].to(device).type(torch.FloatTensor)
    x_val = phi(val_data[:, :2]).to(device)
    y_val = val_data[:, 2:].to(device).type(torch.FloatTensor)
    x_test = phi(test_data[:, :2]).to(device)
    y_test = test_data[:, 2:].to(device).type(torch.FloatTensor)

    if "cuda" in device:
        y_train = y_train.cuda()
        y_val = y_val.cuda()
        y_test = y_test.cuda()

    return x_train, y_train, x_val, y_val, x_test, y_test


def phi(X):
    """
    Augments the 2d X features into a 6D feature set consisting of:
        x_1, x_2, sin(x_1), sin(x_2), x_1^2, x_2^2
    """
    sinX = torch.sin(X)
    squaredX = torch.pow(X, 2)
    return torch.hstack([X, sinX, squaredX])

