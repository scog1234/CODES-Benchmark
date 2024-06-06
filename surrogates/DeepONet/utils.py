from __future__ import annotations

import os
import time
from datetime import datetime
import yaml
from torch import nn
import torch


# TODO complete type hints


def read_yaml_config(model_path):
    config_path = model_path.replace(".pth", ".yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_date_based_directory(base_dir=".", subfolder="models"):
    """
    Create a directory based on the current date (dd-mm format) inside a specified subfolder of the base directory.

    :param base_dir: The base directory where the subfolder and date-based directory will be created.
    :param subfolder: The subfolder inside the base directory to include before the date-based directory.
    :return: The path of the created date-based directory within the specified subfolder.
    """
    # Get the current date in dd-mm format
    current_date = datetime.now().strftime("%m-%d")
    full_path = os.path.join(base_dir, subfolder, current_date)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def save_plot_counter(filename, directory="plots"):
    # Initialize filename and counter
    filepath = os.path.join(directory, filename)
    filebase, fileext = filename.split(".")
    counter = 1

    # Check if the file exists and modify the filename accordingly
    while os.path.exists(filepath):
        filename = f"{filebase}_{counter}.{fileext}"
        filepath = os.path.join(directory, filename)
        counter += 1

    return filepath


def get_project_path(relative_path):
    """
    Construct the absolute path to a project resource (data or model) based on a relative path.

    :param relative_path: A relative path to the resource, e.g., "data/dataset100" or "models/02-28/model.pth".
    :return: The absolute path to the resource.
    """
    import os

    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.realpath(__file__))

    # Navigate up to the parent directory of 'src' and then to the specified relative path
    project_resource_path = os.path.join(current_script_dir, "..", relative_path)

    # Normalize the path to resolve any '..' components
    project_resource_path = os.path.normpath(project_resource_path)

    return project_resource_path


def list_pth_files(directory: str = "models") -> list[str]:
    """
    List all .pth files in the specified directory without their extensions.

    Args:
        directory (str): The directory to search for .pth files.

    Returns:
        list[str]: A list of filenames without the .pth extension.
    """
    files = os.listdir(directory)
    pth_files = [file[:-4] for file in files if file.endswith(".pth")]
    return pth_files


def set_random_seed(gpu_id: int = 0):
    # Combine the current time with the GPU ID to create a unique seed
    seed = int(time.time()) + gpu_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using CUDA


def mass_conservation_loss(
    masses: list,
    criterion=nn.MSELoss(reduction="sum"),
    weights: tuple = (1, 1),
    device: torch.device = torch.device("cpu"),
):
    """
    Replaces the standard MSE loss with a sum of the standard MSE loss and a mass conservation loss.

    :param masses: A list of masses for the chemical species.
    :param criterion: The loss function to use for the standard loss.
    :param weights: A 2-tuple of weights for the standard loss and the mass conservation loss.
    :param device: The device to use for the loss function.

    :return: A new loss function that includes the mass conservation loss.
    """
    masses = torch.tensor(masses, dtype=torch.float32, device=device)

    def loss(outputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Loss function that includes the mass conservation loss.

        :param outputs: The predicted values.
        :param targets: The ground truth values.
        """
        standard_loss = criterion(outputs, targets)

        # Calculate the weighted sum of each chemical quantity for predicted and ground truth,
        # resulting in the total predicted mass and ground truth mass for each sample in the batch
        predicted_mass = torch.sum(outputs * masses, dim=1)
        true_mass = torch.sum(targets * masses, dim=1)

        # Calculate the mass conservation loss as the MSE of the predicted mass vs. true mass
        mass_loss = torch.abs(predicted_mass - true_mass).sum()
        # Sum up the standard MSE loss and the mass conservation loss
        total_loss = weights[0] * standard_loss + weights[1] * mass_loss

        # print(f"Standard loss: {standard_loss.item()}, Mass loss: {mass_loss.item()}")

        return total_loss

    return loss
