import dataclasses
import os
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import create_model_dir


class AbstractSurrogateModel(ABC, nn.Module):
    """
    Abstract base class for surrogate models. This class implements the basic
    structure of a surrogate model and defines the methods that need to be
    implemented by the subclasses for it to be compatible with the benchmarking
    framework. For more information, see
    https://immi000.github.io/CODES-Benchmark-Docs/documentation.html#add_model.

    Args:
        device (str, optional): The device to run the model on. Defaults to None.
        n_chemicals (int, optional): The number of chemicals. Defaults to 29.
        n_timesteps (int, optional): The number of timesteps. Defaults to 100.
        config (dict, optional): The configuration dictionary. Defaults to {}.

    Attributes:
        train_loss (float): The training loss.
        test_loss (float): The test loss.
        MAE (float): The mean absolute error.
        normalisation (dict): The normalisation parameters.
        train_duration (float): The training duration.
        device (str): The device to run the model on.
        n_chemicals (int): The number of chemicals.
        n_timesteps (int): The number of timesteps.
        L1 (nn.L1Loss): The L1 loss function.
        config (dict): The configuration dictionary.

    Methods:

        forward(inputs: Any) -> tuple[Tensor, Tensor]:
            Forward pass of the model.

        prepare_data(
            dataset_train: np.ndarray,
            dataset_test: np.ndarray | None,
            dataset_val: np.ndarray | None,
            timesteps: np.ndarray,
            batch_size: int,
            shuffle: bool,
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
            Gets the data loaders for training, testing, and validation.

        fit(
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int | None,
            position: int,
            description: str,
        ) -> None:
            Trains the model on the training data. Sets the train_loss and test_loss attributes.

        predict(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
            Evaluates the model on the given data loader.

        save(
            model_name: str,
            subfolder: str,
            training_id: str,
            data_params: dict,
        ) -> None:
            Saves the model to disk.

        load(training_id: str, surr_name: str, model_identifier: str) -> None:
            Loads a trained surrogate model.

        setup_progress_bar(epochs: int, position: int, description: str) -> tqdm:
            Helper function to set up a progress bar for training.

        denormalize(data: torch.Tensor) -> torch.Tensor:
            Denormalizes the data back to the original scale.
    """

    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict | None = None,
    ):
        super().__init__()
        self.train_loss = None
        self.test_loss = None
        self.MAE = None
        self.normalisation = None
        self.device = device
        self.n_chemicals = n_chemicals
        self.n_timesteps = n_timesteps
        self.L1 = nn.L1Loss()
        self.config = config if config is not None else {}
        self.train_duration = None

    @abstractmethod
    def forward(self, inputs: Any) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (Any): The input data as recieved from the dataloader.

        Returns:
            tuple[Tensor, Tensor]: The model predictions and the targets.
        """
        pass

    @abstractmethod
    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Prepare the data for training, testing, and validation. This method should
        return the DataLoader objects for the training, testing, and validation data.

        Args:
            dataset_train (np.ndarray): The training dataset.
            dataset_test (np.ndarray): The testing dataset.
            dataset_val (np.ndarray): The validation dataset.
            timesteps (np.ndarray): The timesteps.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: The DataLoader objects for the
                training, testing, and validation data.
        """
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int,
        description: str,
    ) -> None:
        """
        Perform the training of the model. Sets the train_loss and test_loss attributes.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the testing data.
            epochs (int): The number of epochs to train the model for.
            position (int): The position of the progress bar.
            description (str): The description of the progress bar.
        """
        pass

    def predict(
        self,
        data_loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the model on the given dataloader.

        Args:
            data_loader (DataLoader): The DataLoader object containing the data the
                model is evaluated on.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predictions and targets.
        """

        # infer output size
        with torch.inference_mode():
            dummy_inputs = next(iter(data_loader))
            dummy_outputs, _ = self.forward(dummy_inputs)
            batch_size, out_shape = (
                dummy_outputs.shape[0],
                dummy_outputs.shape[-(dummy_outputs.ndim - 1) :],
            )

        # pre-allocate buffers for predictions and targets
        size = (batch_size * len(data_loader), *out_shape)
        predictions = torch.zeros(size, dtype=dummy_outputs.dtype).to(self.device)
        targets = torch.zeros(size, dtype=dummy_outputs.dtype).to(self.device)

        processed_samples = 0

        with torch.inference_mode():
            for i, inputs in enumerate(data_loader):
                preds, targs = self.forward(inputs)
                batch_size = preds.shape[0]
                predictions[i * batch_size : (i + 1) * batch_size, ...] = preds
                targets[i * batch_size : (i + 1) * batch_size, ...] = targs
                processed_samples += batch_size

        # Slice the buffers to include only the processed samples
        predictions = predictions[:processed_samples, ...]
        targets = targets[:processed_samples, ...]

        predictions = self.denormalize(predictions)
        targets = self.denormalize(targets)

        predictions = predictions.reshape(-1, self.n_timesteps, self.n_chemicals)
        targets = targets.reshape(-1, self.n_timesteps, self.n_chemicals)

        return predictions, targets

    def save(
        self,
        model_name: str,
        base_dir: str,
        training_id: str,
        data_params: dict,
    ) -> None:
        """
        Save the model to disk.

        Args:
            model_name (str): The name of the model.
            subfolder (str): The subfolder to save the model in.
            training_id (str): The training identifier.
            data_params (dict): The data parameters.
        """

        # Make the model directory
        subfolder = os.path.join(base_dir, training_id, self.__class__.__name__)
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
            "n_train_samples",
            "n_timesteps",
        ]
        for attr in check_attributes:
            if hasattr(self, attr):
                hyperparameters[attr] = getattr(self, attr)
                # n_timesteps must not be saved to the model, it will cause problems in the benchmark
                if attr == "n_timesteps":
                    delattr(self, attr)

        # Add some additional information to the model and hyperparameters
        self.train_duration = (
            self.fit.duration if hasattr(self.fit, "duration") else None
        )
        hyperparameters["train_duration"] = self.train_duration
        self.normalisation = data_params
        hyperparameters["normalisation"] = data_params

        # Reduce the precision of the losses and accuracy
        for attribute in ["train_loss", "test_loss", "MAE"]:
            value = getattr(self, attribute)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.cpu().detach().numpy()
                if isinstance(value, np.ndarray):
                    value = value.astype(np.float16)
                setattr(self, attribute, value)

        # Save the hyperparameters as a yaml file
        hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
        with open(hyperparameters_path, "w", encoding="utf-8") as file:
            yaml.dump(hyperparameters, file)

        save_attributes = {
            k: v
            for k, v in self.__dict__.items()
            if k != "state_dict" and not k.startswith("_")
        }
        model_dict = {"state_dict": self.state_dict(), "attributes": save_attributes}

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model_dict, model_path)

    def load(
        self,
        training_id: str,
        surr_name: str,
        model_identifier: str,
        model_dir: str | None = None,
    ) -> None:
        """
        Load a trained surrogate model.

        Args:
            training_id (str): The training identifier.
            surr_name (str): The name of the surrogate model.
            model_identifier (str): The identifier of the model (e.g., 'main').

        Returns:
            None. The model is loaded in place.
        """
        if model_dir is None:
            model_dict_path = os.path.join(
                os.getcwd(), "trained", training_id, surr_name, f"{model_identifier}.pth"
            )
        else:
            model_dict_path = os.path.join(
                model_dir, training_id, surr_name, f"{model_identifier}.pth"
            )
        model_dict = torch.load(model_dict_path, map_location=self.device)
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
        """
        Helper function to set up a progress bar for training.

        Args:
            epochs (int): The number of epochs.
            position (int): The position of the progress bar.
            description (str): The description of the progress bar.

        Returns:
            tqdm: The progress bar.
        """
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
                ...
            elif self.normalisation["mode"] == "minmax":
                dmax = self.normalisation["max"]
                dmin = self.normalisation["min"]
                data = (data + 1) * (dmax - dmin) / 2 + dmin
            elif self.normalisation["mode"] == "standardize":
                mean = self.normalisation["mean"]
                std = self.normalisation["std"]
                data = data * std + mean

        return data


SurrogateModel = TypeVar("SurrogateModel", bound=AbstractSurrogateModel)
