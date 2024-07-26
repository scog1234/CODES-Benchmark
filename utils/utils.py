import yaml
import functools
import time
import os
import shutil
import random
import numpy as np
import torch


def read_yaml_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.duration = end_time - start_time
        print(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def create_model_dir(
    base_dir: str = ".", subfolder: str = "trained", unique_id: str = ""
) -> str:
    """
    Create a directory based on a unique identifier inside a specified subfolder of the base directory.

    :param base_dir: The base directory where the subfolder and unique directory will be created.
    :param subfolder: The subfolder inside the base directory to include before the unique directory.
    :param unique_id: A unique identifier to be included in the directory name.
    :return: The path of the created unique directory within the specified subfolder.
    """
    full_path = os.path.join(base_dir, subfolder, unique_id)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def load_and_save_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file and save a copy to the specified directory.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    # Load configuration from YAML
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get training ID from the config
    training_id = config["training_id"]

    # Create the directory if it does not exist
    save_dir = os.path.join("trained", training_id)
    os.makedirs(save_dir, exist_ok=True)

    # Copy the config file to the directory
    config_save_path = os.path.join(save_dir, "config.yaml")
    shutil.copyfile(config_path, config_save_path)

    return config


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nice_print(message: str, width: int = 80) -> None:
    """
    Print a message in a nicely formatted way with a fixed width.

    Args:
        message (str): The message to print.
        width (int): The width of the printed box. Default is 80.
    """
    # Calculate padding
    padding = (width - len(message) - 2) // 2
    padding_left = padding
    padding_right = padding

    # If message length is odd, add one more space to the right
    if (width - len(message)) % 2 != 0:
        padding_right += 1

    border = "-" * width
    print(
        f"\n{border}\n|{' ' * padding_left}{message}{' ' * padding_right}|\n{border}\n"
    )


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    np.random.seed(np_seed)
