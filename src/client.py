import gc
import logging

import torch
# import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_target_delta, EPSILON


logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.max_eps = -1
        self.__model = None
        self.delta = 0

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(
            self.data, batch_size=client_config["batch_size"], shuffle=True
        )
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.task = client_config["task"]
        self.labels = client_config["labels"]
        self.mgn = client_config["mgn"]
        self.is_dp = client_config["is_dp"]
        self.delta = get_target_delta(len(self.dataloader))
        print(f"Setup client {self.id} with delta = {self.delta}")

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)

        if self.is_dp:
            privacy_engine = PrivacyEngine()

            model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                epochs=self.local_epoch,
                target_epsilon=EPSILON,
                target_delta=self.delta,
                max_grad_norm=self.mgn,
            )

            print(f"Using sigma={optimizer.noise_multiplier} and C={self.mgn}")

            for e in range(self.local_epoch):
                with BatchMemoryManager(
                    data_loader=dataloader,
                    max_physical_batch_size=16,
                    optimizer=optimizer,
                ) as memory_safe_data_loader:
                    for i, (data, labels) in enumerate(tqdm(memory_safe_data_loader)):
                        data, labels = data.to(self.device), labels.to(self.device)

                        optimizer.zero_grad()
                        outputs = model(data)
                        if self.task == "multi-label, binary-class":
                            labels = labels.to(torch.float32)
                        else:
                            labels = labels.squeeze().long()
                        if not labels.shape:
                            continue

                        loss = eval(self.criterion)()(outputs, labels)

                        loss.backward()
                        optimizer.step()

                        if self.device == "cuda":
                            torch.cuda.empty_cache()

                self.max_eps = max(self.max_eps, privacy_engine.get_epsilon(self.delta))
                for k, v in model.state_dict().items():
                    renamed_k = k.replace("_module.", "")
                    self.model.state_dict()[renamed_k] = v
        else:
            for e in range(self.local_epoch):
                for i, (data, labels) in enumerate(tqdm(self.dataloader)):
                    data, labels = data.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(data)
                    if self.task == "multi-label, binary-class":
                        labels = labels.to(torch.float32)
                    else:
                        labels = labels.squeeze().long()
                    if not labels.shape:
                        continue

                    loss = eval(self.criterion)()(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    if self.device == "cuda":
                        torch.cuda.empty_cache()
        self.model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
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

                if self.device == "cuda":
                    torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader) - invalid_data_count
        test_accuracy = correct / len(self.data) - invalid_data_count

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True)
        logging.info(message)
        del message
        gc.collect()

        return test_loss, test_accuracy
