import os
import logging

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from medmnist import INFO
import matplotlib.pyplot as plt

from torch.utils.data import Subset
import torchvision

logger = logging.getLogger(__name__)
EPSILON = 3


def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True


def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    f"[ERROR] ...initialization method [{init_type}] is not implemented!"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif (
            classname.find("BatchNorm2d") != -1
            or classname.find("InstanceNorm2d") != -1
        ):
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert (
            torch.cuda.is_available()
        ), ("CUDA is not available. Make sure your system has a compatible GPU and that you have "
            "installed the necessary CUDA drivers and PyTorch with GPU support.")
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model


def create_splits(number: int, split: int = 11, ratio: float = 0.75, seed: int = 42):
    """Function used to create splits for federated datasets.

    Args:
        number: The number to split.
        split: Number of splits to do.
        ratio: Determine the minimum and maximum size of one split.
        seed: Random seed.

    Returns:
        splits: The list of elements per split.
        added_splits: Number of elements per splits matched to index.
    """

    # Set the seed to always get the same splits for evaluation purposes
    np.random.seed(seed)
    # Contains number of elements per split
    splits = []
    # Contains cumulated sum of splits to match indexes
    added_splits = []
    entire_part = number // split
    # A single split cannot be lower than entire_part - min_split
    min_split = entire_part * ratio
    if number < split:
        return [number]
    for i in range(split):
        if number % split != 0 and i >= split - (number % split):
            splits.append(entire_part + 1)
        else:
            splits.append(entire_part)
    length = len(splits) if len(splits) % 2 == 0 else len(splits) - 1
    for s in range(0, length, 2):
        random_value = np.random.randint(low=0, high=min_split)
        splits[s] -= random_value
        added_splits.append(int(np.sum(splits[:s])))
        splits[s + 1] += random_value
        added_splits.append(int(np.sum(splits[: s + 1])))
    if len(splits) % 2 != 0:
        added_splits.append(np.sum(splits[:-1]))
    added_splits.append(np.sum(splits))
    return splits, added_splits


def create_datasets(data_path, dataset_name, num_clients, save_data_distribution=False):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    if dataset_name not in INFO.keys():
        # dataset not found exception
        error_message = f'Dataset "{dataset_name}" is not supported or cannot be found in TorchVision Datasets!'
        raise AttributeError(error_message)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    info = INFO[dataset_name]
    task = info["task"]
    DataClass = getattr(medmnist, info["python_class"])

    training_dataset = DataClass(
        root=data_path, split="train", transform=transform, download=True
    )

    test_dataset = DataClass(
        root=data_path, split="test", transform=transform, download=True
    )

    _, added_splits = create_splits(
        len(training_dataset), split=num_clients, ratio=0.50
    )
    # Create subset
    local_datasets = []
    local_datasets_len = []
    for c in range(num_clients):
        client_dataset = Subset(
            training_dataset,
            range(int(added_splits[c]), int(added_splits[c + 1])),
        )
        local_datasets.append(client_dataset)
        local_datasets_len.append(len(client_dataset))
    if save_data_distribution:
        save_data_distribution_among_clients(num_clients, local_datasets_len, info["python_class"])
    labels = [*info["label"].values()]
    return task, local_datasets, test_dataset, labels


def save_data_distribution_among_clients(num_clients: int, data_length: list, dataset_name: str):
    plt.style.use("ggplot")
    plt.rc("font", family="serif")

    plt.figure(figsize=(8, 6))

    # Use a horizontal bar plot
    plt.bar(
        range(num_clients),
        data_length,
        width=0.8,
        color="#8B008B",
        linewidth=1,
        linestyle="-.",
    )

    # Add data length labels over each bar
    for i, data in enumerate(data_length):
        plt.text(i, data + 10, str(data), va="center", fontsize=12, color="black")

    # Set x-axis ticks to represent clients
    plt.xticks(range(num_clients), [i + 1 for i in range(num_clients)])

    # Add a title and labels
    plt.title(f"{dataset_name} dataset distribution among {num_clients} hospitals")

    plt.ylabel("Data size")
    plt.xlabel("Participant No.")

    plt.savefig(f"{num_clients}_clients_{dataset_name}_data_distribution.png")


def get_target_delta(data_size: int) -> float:
    """Generate target delta given the size of a dataset. Delta should be
    less than the inverse of the datasize.

    Parameters
    ----------
    data_size : int
        The size of the dataset.

    Returns
    -------
    float
        The target delta value.
    """
    den = 1
    while data_size // den >= 1:
        den *= 10
    return 1 / den


def get_datasets(data_path: str, dataset_name: str) -> tuple:
    """Loads needed dataset's training and test data

    Returns:
        task type, training data, test data, labels
    """
    if dataset_name not in INFO.keys():
        # dataset not found exception
        error_message = f"...dataset ({dataset_name}) is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    info = INFO[dataset_name]
    task = info["task"]
    DataClass = getattr(medmnist, info["python_class"])

    training_dataset = DataClass(
        root=data_path, split="train", transform=transform, download=True
    )

    test_dataset = DataClass(
        root=data_path, split="test", transform=transform, download=True
    )
    labels = [*info["label"].values()]
    return task, training_dataset, test_dataset, labels


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            raise RuntimeError(f"Failed to create the folder '{folder_path}'. Error: {str(e)}")
    else:
        print(f"Folder '{folder_path}' already exists.")