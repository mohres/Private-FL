import re
import time
import datetime
import pickle

import yaml
import threading
import logging
import copy

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.traditional_server import TraditionalServer
from src.utils import launch_tensor_board

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:300.0"


colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["o", "s", "D", "^", "v", "*", "P"]
plt.style.use("ggplot")
plt.rc("font", family="serif")


def setup_and_train_traditional():
    # read configuration file
    with open("./config.yaml") as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    traditional_config = configs[7]["traditional_config"]

    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(
        log_config["log_path"],
        str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
    )

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]]),
    ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p",
    )
    plt.ioff()
    fig, ax = plt.subplots()
    for idx, dataset in enumerate(data_config["datasets_names"]):
        # plt.ioff()
        # display and log experiment configuration
        message = "\n[WELCOME] Unfolding configurations...!"
        print(message)
        logging.info(message)

        for config in configs:
            print(config)
            logging.info(config)
        print()
        # initialize federated learning
        data_config_ = copy.deepcopy(data_config)
        model_config_ = copy.deepcopy(model_config)
        data_config_["dataset_name"] = dataset
        model_config_["in_channels"], model_config_["num_classes"] = (
            model_config["channels"][idx],
            model_config["classes"][idx],
        )
        del (
            data_config_["datasets_names"],
            model_config_["channels"],
            model_config_["classes"],
        )

        central_server = TraditionalServer(
            writer,
            model_config_,
            global_config,
            data_config_,
            init_config,
            traditional_config,
            optim_config,
        )

        central_server.setup()

        central_server.fit()

        # save resulting losses and metrics

        with open(os.path.join(
                log_config["log_path"],
                f"{model_config_['name']}_{central_server.dataset_name}_traditional_results.pkl"
        ), "wb") as f:
            pickle.dump(central_server.results, f)

        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Tradition Training on {central_server.dataset_name} Dataset"
        )
        smoothing_factor = 0.1
        # Applying smoothing to the accuracy values
        smoothed_accuracy = [central_server.results["accuracy"][0]]
        for value in central_server.results["accuracy"][1:]:
            smoothed_value = (
                    smoothed_accuracy[-1] * (1 - smoothing_factor) + value * smoothing_factor
            )
            smoothed_accuracy.append(smoothed_value)

        # Plotting the smoothed curve
        ax.plot(
            range(central_server.num_rounds),
            smoothed_accuracy,
            color=colors[idx],
            # linestyle='dashed',
            label=f"{central_server.dataset_name}",
        )
        ax.set_xlim(-0.5, central_server.num_rounds + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.savefig(
            f"Accuracy_"
            f"_{model_config_['name']}"
            f"_R:{traditional_config['R']}"
            f"_B:{traditional_config['B']}"
            f"_LR:{optim_config['lr']}"
            f"_traditional.png"
        )


def setup_and_train_fed():
    # read configuration file
    with open("./config.yaml") as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(
        log_config["log_path"],
        str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
    )

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]]),
    ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p",
    )

    for idx, dataset in enumerate(data_config["datasets_names"]):
        fig, ax = plt.subplots()
        # plt.ioff()
        for c_idx, client_size in enumerate(fed_config["K"]):
            # display and log experiment configuration
            message = "\n[WELCOME] Unfolding configurations...!"
            print(message)
            logging.info(message)

            for config in configs:
                print(config)
                logging.info(config)
            print()
            # initialize federated learning
            data_config_ = copy.deepcopy(data_config)
            model_config_ = copy.deepcopy(model_config)
            fed_config_ = copy.deepcopy(fed_config)
            data_config_["dataset_name"] = dataset
            model_config_["in_channels"], model_config_["num_classes"] = (
                model_config["channels"][idx],
                model_config["classes"][idx],
            )
            fed_config_["K"] = client_size
            del (
                data_config_["datasets_names"],
                model_config_["channels"],
                model_config_["classes"],
            )
            central_server = Server(
                writer,
                model_config_,
                global_config,
                data_config_,
                init_config,
                fed_config_,
                optim_config,
            )

            central_server.setup()

            central_server.fit()

            # save resulting losses and metrics
            with open(
                os.path.join(
                    log_config["log_path"],
                    f"{model_config_['name']}_{central_server.dataset_name}_{client_size}_results.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(central_server.results, f)

            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            # ax.set_title(
            #     f"{central_server.dataset_name} Dataset"
            # )
            ax.plot(
                range(central_server.num_rounds),
                central_server.accuracies,
                # marker=markers[c_idx],
                color=colors[c_idx],
                label=f"{client_size}",
            )
            ax.set_xlim(-0.5, central_server.num_rounds + 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.legend()
            fig.savefig(
                f"Accuracy_{central_server.dataset_name}"
                f"_{model_config_['name']}"
                f"_R:{fed_config_['R']}"
                f"_E:{fed_config_['E']}"
                f"_B:{fed_config_['B']}"
                f"_traditional.png"
            )
        plt.ioff()


def extract_number_from_filename(filename):
    if num := int(filename.split("_")[2]):
        return num
    return None


def plot_smoothed_accuracy_from_pkl_traditional(pkl_dir_path, name_pattern, smoothing_factor=0.1):
    # Get a list of all pickle files in the specified directory
    pkl_files = [
        filename for filename in os.listdir(pkl_dir_path) if
        filename.startswith(name_pattern) and filename.endswith("_results.pkl")
    ]

    # Create a new figure and axis
    fig, ax = plt.subplots()

    for idx, pkl_file in enumerate(pkl_files):
        # Extract the name from the file name (e.g., "ResNet9_bloodmnist_traditional_results.pkl" -> 1)
        # Load data from the pickle file
        pkl_file_path = os.path.join(pkl_dir_path, pkl_file)
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        accuracy_values = data["accuracy"]
        print(f"{pkl_file} : {max(accuracy_values)}")
        continue

        # Smoothing the accuracy data using a simple moving average
        smoothed_accuracy = [accuracy_values[0]]
        for value in accuracy_values[1:]:
            smoothed_value = (
                    smoothed_accuracy[-1] * (1 - smoothing_factor) + value * smoothing_factor
            )
            smoothed_accuracy.append(smoothed_value)

        # ax.set_title(f"Accuracy Plot for ResNet_bloodmnist_{number}")
        ax.plot(
            range(len(accuracy_values)),
            smoothed_accuracy,
            color=colors[6],
            # label=f"{dataset_name}",
        )
        ax.set_xlim(-0.5, len(accuracy_values) + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        # ax.legend()

        fig.savefig(f"Accuracy_smoothed_resnet9_{name_pattern}.png")


def plot_smoothed_accuracy_from_pkl(pkl_dir_path, name_pattern, smoothing_factor=0.1, special_name=""):
    # Get a list of all pickle files in the specified directory
    pkl_files = [
        filename for filename in os.listdir(pkl_dir_path) if
        filename.startswith(name_pattern) and filename.endswith("_results.pkl")
    ]
    pkl_files.sort(key=lambda x: extract_number_from_filename(x))

    # Create a new figure and axis
    fig, ax = plt.subplots()

    for idx, pkl_file in enumerate(pkl_files):
        # Extract the number from the file name (e.g., "ResNet_bloodmnist_2_results.pkl" -> 2)
        number = extract_number_from_filename(pkl_file)

        # Load data from the pickle file
        pkl_file_path = os.path.join(pkl_dir_path, pkl_file)
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        accuracy_values = data["accuracy"]
        print(f"{pkl_file} - {number} : {max(accuracy_values)}")

        # Smoothing the accuracy data using a simple moving average
        smoothed_accuracy = [accuracy_values[0]]
        for value in accuracy_values[1:]:
            smoothed_value = (
                    smoothed_accuracy[-1] * (1 - smoothing_factor) + value * smoothing_factor
            )
            smoothed_accuracy.append(smoothed_value)

        # ax.set_title(f"Accuracy Plot for ResNet_bloodmnist_{number}")
        ax.plot(
            range(len(accuracy_values)),
            smoothed_accuracy,
            color=colors[idx],
            label=f"{number} H",
        )
        ax.set_xlim(-0.5, len(accuracy_values) + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.legend()

        fig.savefig(f"Accuracy_smoothed_{name_pattern}_{special_name}.png")
    plt.close(fig)  # Close the figure to avoid displaying multiple plots


if __name__ == "__main__":
    # setup_and_train_traditional()
    setup_and_train_fed()
    message = "[Done] all learning process!\n...exit program!"
    print(message)
    logging.info(message)
    time.sleep(3)
    exit()

