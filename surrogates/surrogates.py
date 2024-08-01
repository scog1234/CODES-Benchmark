from abc import ABC, abstractmethod
from typing import TypeVar
import os
import dataclasses
import yaml

# from typing import Optional, Union

import torch
from tqdm import tqdm
from torch import nn, Tensor
from torch.utils.data import DataLoader
import numpy as np

from utils import create_model_dir


# Define abstract base class for surrogate models
class AbstractSurrogateModel(ABC, nn.Module):

    def __init__(self, device: str | None = None, N_chemicals: int = 29):
        super().__init__()
        self.train_loss = None
        self.test_loss = None
        self.accuracy = None
        self.normalisation = None
        self.device = device
        self.N_chemicals = N_chemicals

    @abstractmethod
    def forward(self, inputs, timesteps: np.ndarray) -> Tensor:
        pass

    @abstractmethod
    def prepare_data(
        self,
        timesteps: np.ndarray,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None = None,
        dataset_val: np.ndarray | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader | Tensor,
        test_loader: DataLoader | Tensor,
        timesteps: np.ndarray,
        epochs: int | None,
        position: int,
        description: str,
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        data_loader: DataLoader | Tensor,
        timesteps: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        pass

    def save(
        self,
        model_name: str,
        subfolder: str,
        training_id: str,
        data_params: dict,
    ) -> None:

        # Make the model directory
        base_dir = os.getcwd()
        subfolder = os.path.join(subfolder, training_id, self.__class__.__name__)
        model_dir = create_model_dir(base_dir, subfolder)

        # Load and clean the hyperparameters
        hyperparameters = dataclasses.asdict(self.config)
        remove_keys = ["masses"]
        for key in remove_keys:
            hyperparameters.pop(key, None)
        for key in hyperparameters.keys():
            if isinstance(hyperparameters[key], nn.Module):
                hyperparameters[key] = hyperparameters[key].__class__.__name__

        # Check if the model has some attributes. If so, add them to the hyperparameters
        check_attributes = [
            "N_train_samples",
            "N_timesteps",
        ]
        for attr in check_attributes:
            if hasattr(self, attr):
                hyperparameters[attr] = getattr(self, attr)

        # Add some additional information to the model and hyperparameters
        self.train_duration = self.fit.duration
        hyperparameters["train_duration"] = self.train_duration
        setattr(self, "normalisation", data_params)
        hyperparameters["normalisation"] = data_params

        # Reduce the precision of the losses and accuracy
        for attribute in ["train_loss", "test_loss", "accuracy"]:
            value = getattr(self, attribute)
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy()
            if value is not None:
                value = value.astype(np.float16)
                setattr(self, attribute, value)

        # Save the hyperparameters as a yaml file
        hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
        with open(hyperparameters_path, "w") as file:
            yaml.dump(hyperparameters, file)

        save_attributes = {
            k: v
            for k, v in self.__dict__.items()
            if k != "state_dict" and not k.startswith("_")
        }
        model_dict = {"state_dict": self.state_dict(), "attributes": save_attributes}

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model_dict, model_path)

        # tqdm.write(f"Model, losses and hyperparameters saved to {model_dir}")

    def load(self, training_id: str, surr_name: str, model_identifier: str) -> None:
        """
        Load a trained surrogate model.

        Args:
            model: Instance of the surrogate model class.
            training_id (str): The training identifier.
            surr_name (str): The name of the surrogate model.
            model_identifier (str): The identifier of the model (e.g., 'main').

        Returns:
            None. Tje model is loaded in place.
        """
        model_dict_path = os.path.join(
            "trained", training_id, surr_name, f"{model_identifier}.pth"
        )
        model_dict = torch.load(model_dict_path)
        self.load_state_dict(model_dict["state_dict"])
        for key, value in model_dict["attributes"].items():
            # remove self.device from the attributes
            if key == "device":
                continue
            else:
                setattr(self, key, value)
        self.to(self.device)
        self.eval()

    def setup_progress_bar(self, epochs: int, position: int, description: str):
        bar_format = "{l_bar}{bar}| {n_fmt:>5}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
        progress_bar = tqdm(
            range(epochs),
            desc=description,
            position=position,
            leave=False,
            bar_format=bar_format,
        )
        progress_bar.set_postfix(
            {"loss": f"{0:.2e}", "lr": f"{self.config.learning_rate:.1e}"}
        )
        return progress_bar

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the data.

        Args:
            data (np.ndarray): The data to denormalize.

        Returns:
            np.ndarray: The denormalized data.
        """
        if self.normalisation is not None:
            if self.normalisation["mode"] == "disabled":
                data = data
            elif self.normalisation["mode"] == "minmax":
                dmax = self.normalisation["max"]
                dmin = self.normalisation["min"]
                data = (data + 1) * (dmax - dmin) / 2 + dmin
            elif self.normalisation["mode"] == "standardize":
                mean = self.normalisation["mean"]
                std = self.normalisation["std"]
                data = data * std + mean

            if self.normalisation["log10_transform"]:
                data = 10**data

        return data


SurrogateModel = TypeVar("SurrogateModel", bound=AbstractSurrogateModel)
