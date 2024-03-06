import copy
import gc
from collections import OrderedDict

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import *
from .utils import *
from .client import Client
from sklearn.metrics import confusion_matrix
import seaborn as sns

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning

    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will receive the updated global model as its local model.

    Attributes:
        writer (SummaryWriter): instance to track a metric and a loss of the global model.
        model_config: kwargs related to the model
        global_config: for global settings
        data_config: kwargs related to dataset
        init_config: kwargs for the initialization of the model.
        fed_config: Kwargs for federated average algorithm.
        optim_config: Kwargs provided for optimizer.
    """

    def __init__(
        self,
        writer,
        model_config={},
        global_config={},
        data_config={},
        init_config={},
        fed_config={},
        optim_config={},
    ):
        self.dataloader = None
        self.test_data = None
        self.labels = None
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config["name"])(**model_config)

        self.seed: int = global_config["seed"]
        # Training machine indicator (e.g. "cpu", "cuda").
        self.device: str = global_config["device"]
        # indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        self.mp_flag: bool = global_config["is_mp"]
        # is the training is differentially private
        self.is_dp: bool = global_config["is_dp"]
        self.save_cm: bool = global_config["save_cm"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.task = data_config["task"]
        self.mgn = data_config["mgn"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        self.accuracies = []
        self.losses = []

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # create root file if not exist
        create_folder(self.data_path)
        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, **self.init_config)

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Successfully initialized model "
                   f"(# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

        # split local dataset for each client

        task, local_datasets, test_dataset, labels = create_datasets(
            self.data_path,
            self.dataset_name,
            self.num_clients
        )
        self.labels = labels

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.test_data = test_dataset
        self.dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # configure detailed settings for client update and
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion,
            num_local_epochs=self.local_epochs,
            optimizer=self.optimizer,
            optim_config=self.optim_config,
            task=task,
            labels=labels,
            mgn=self.mgn,
            is_dp=self.is_dp,
        )

        # send the model skeleton to all clients
        self.transmit_model()

    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Successfully created all {str(self.num_clients)} clients!")
        print(message)
        logging.info(message)
        del message
        gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Successfully finished setup of all {str(self.num_clients)} clients!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = (f"[Round: {str(self._round).zfill(4)}] "
                       f"Successfully transmitted models to all {str(self.num_clients)} clients!")
            print(message)
            logging.info(message)
            del message
            gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)

            message = (f"[Round: {str(self._round).zfill(4)}] "
                       f"Successfully transmitted models to {str(len(sampled_client_indices))} selected clients!")
            print(message)
            logging.info(message)
            del message
            gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randomly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message)
        logging.info(message)
        del message
        gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(
                a=[i for i in range(self.num_clients)],
                size=num_sampled_clients,
                replace=False,
            ).tolist()
        )

        return sampled_client_indices

    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Start updating selected {len(sampled_client_indices)} clients...!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"{len(sampled_client_indices)} clients are selected and updated "
                   f"(with total sample size: {str(selected_total_size)})!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

        return selected_total_size

    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True)
        logging.info(message)
        del message
        gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Client {str(self.clients[selected_index].id).zfill(4)} is selected and updated "
                   f"(with total sample size: {str(client_size)})!")
        print(message, flush=True)
        logging.info(message)
        del message
        gc.collect()

        return client_size

    def average_model(self, sampled_client_indices: list, coefficients: list):
        """Average the updated and transmitted parameters from each selected client."""
        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Aggregate updated weights of {len(sampled_client_indices)} clients...!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Updated weights of {len(sampled_client_indices)} clients are successfully averaged!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Evaluate selected {str(len(sampled_client_indices))} clients' models...!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate(self._round)

        message = (f"[Round: {str(self._round).zfill(4)}] "
                   f"Finished evaluation of {str(len(sampled_client_indices))} selected clients!")
        print(message)
        logging.info(message)
        del message
        gc.collect()

    def mp_evaluate_selected_models(self, selected_index: int):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(
                    self.mp_update_selected_clients, sampled_client_indices
                )
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = (f"[Round: {str(self._round).zfill(4)}] "
                       f"Evaluate selected {str(len(sampled_client_indices))} clients' models...!")
            print(message)
            logging.info(message)
            del message
            gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.test_data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        all_true_labels = []
        all_predicted_labels = []
        with torch.no_grad():
            invalid_data_count = 0
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                outputs = self.model(data)
                if self.task == "multi-label, binary-class":
                    labels = labels.to(torch.float32)
                else:
                    labels = labels.squeeze().long()
                if not labels.shape:
                    invalid_data_count += 1
                    continue
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())

                if self.device == "cuda":
                    torch.cuda.empty_cache()
        self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader) - invalid_data_count
        test_accuracy = correct / len(self.test_data) - invalid_data_count

        cm = confusion_matrix(all_true_labels, all_predicted_labels)

        def wrap_labels(max_chars_per_line=20):
            wrapped_labels = []
            for label in self.labels:
                words = label.split()  # Split the label into words
                lines = []
                current_line = words[0]

                for word in words[1:]:
                    if len(current_line) + len(word) + 1 <= max_chars_per_line:
                        current_line += " " + word  # Add the word to the current line
                    else:
                        lines.append(current_line)  # Start a new line
                        current_line = word

                lines.append(current_line)  # Add the last line

                # Join the lines with a newline character
                wrapped_label = "-\n".join(lines)
                wrapped_labels.append(wrapped_label)

            return wrapped_labels

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=wrap_labels(),
            yticklabels=wrap_labels(),
        )

        # Rotate the tick labels for better visibility
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            f"{self.dataset_name.capitalize()} Confusion Matrix - Round {self._round} ({self.num_clients} Clients)"
        )

        plt.subplots_adjust(top=0.95, right=1, left=0.2, bottom=0.25)
        # Save the confusion matrix as an image
        if self.save_cm:
            plt.savefig(
                f"{self.dataset_name}_round_{self._round}_client_{self.num_clients}_confusion_matrix.png"
            )

        # Close the plot to release resources
        plt.close()

        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            self.results["loss"].append(test_loss)
            self.results["accuracy"].append(test_accuracy)

            log_details = (
                f"{self.dataset_name}"
                f"_{self.model.name} "
                f"C_{self.fraction}, "
                f"K_{self.num_clients}, "
                f"E_{self.local_epochs}, "
                f"B_{self.batch_size}, "
                f"lr_{self.optim_config['lr']}"
            )
            if self.is_dp:
                log_details += f"_EPS: {EPSILON}, MGN: {self.mgn}"

            self.losses.append(test_loss)
            self.accuracies.append(test_accuracy)

            self.writer.add_scalars(
                "Loss",
                {log_details: test_loss},
                self._round,
            )
            self.writer.add_scalars(
                "Accuracy",
                {log_details: test_accuracy},
                self._round,
            )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}\
                \n"
            if not self.is_dp:
                message += f"\t=> eps: {[(client.id, client.max_eps) for client in self.clients]}"
            print(message)
            logging.info(message)
            del message
            gc.collect()

        self.transmit_model()
