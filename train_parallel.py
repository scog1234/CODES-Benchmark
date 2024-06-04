import yaml
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from surrogates.surrogate_classes import surrogate_classes
from data import check_and_load_data

# Load configuration from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_data_subset(full_train_data, full_test_data, osu_timesteps, mode, metric):
    """
    Get the appropriate data subset based on the mode and metric.

    Args:
        full_train_data (np.ndarray): The full training data.
        full_test_data (np.ndarray): The full test data.
        osu_timesteps (np.ndarray): The timesteps.
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor).

    Returns:
        tuple: The training data, test data, and timesteps subset.
    """
    if mode == "interpolation":
        interval = int(metric)
        train_data = full_train_data[:, ::interval]
        test_data = full_test_data[:, ::interval]
        timesteps = osu_timesteps[::interval]
    elif mode == "extrapolation":
        cutoff = int(metric)
        train_data = full_train_data[:, :cutoff]
        test_data = full_test_data[:, :cutoff]
        timesteps = osu_timesteps[:cutoff]
    elif mode == "sparse":
        factor = int(metric)
        train_data = full_train_data[::factor]
        test_data = full_test_data[::factor]
        timesteps = osu_timesteps
    else:
        train_data = full_train_data
        test_data = full_test_data
        timesteps = osu_timesteps

    return train_data, test_data, timesteps


def train_and_save_model(
    mode: str,
    surrogate_name: str,
    metric: str,
    surrogate_class,
    config,
    device: str,
):
    """
    Train and save a model for a specific benchmark mode.

    Args:
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        surrogate_name (str): The name of the surrogate model.
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor).
        surrogate_class: The class of the surrogate model.
        config (dict): The configuration dictionary.
        device (str): The device to use for training (e.g., 'cuda:0').
    """
    # Set the device for the model
    os.environ["CUDA_VISIBLE_DEVICES"] = device if "cuda" in device else ""
    model = surrogate_class(device=device)

    # Load full data
    full_train_data, osu_timesteps, _ = check_and_load_data(config["dataset"], "train")
    full_test_data, _, _ = check_and_load_data(config["dataset"], "test")

    # Get the appropriate data subset
    train_data, test_data, timesteps = get_data_subset(
        full_train_data, full_test_data, osu_timesteps, mode, metric
    )

    # Train the model
    model.fit(train_data, test_data, timesteps)

    # Save the model (making the name lowercase and removing any underscores)
    model_name = f"{surrogate_name.lower()}_{mode}_{metric}".strip("_")
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        subfolder=f"trained/{surrogate_name}",
        unique_id=config["training_id"],
        dataset_name=config["dataset"],
    )


def train_surrogate(config, surrogate_class, surrogate_name):
    """
    Train and save models for different purposes based on the config settings.

    Args:
        config (dict): The configuration dictionary.
        surrogate_class: The class of the surrogate model to train.
        surrogate_name (str): The name of the surrogate model.
    """

    tasks = []

    if config["accuracy"]:
        tasks.append(("main", surrogate_name, "", surrogate_class, config))

    if config["interpolation"]["enabled"]:
        for interval in config["interpolation"]["intervals"]:
            tasks.append(
                (
                    "interpolation",
                    surrogate_name,
                    str(interval),
                    surrogate_class,
                    config,
                )
            )

    if config["extrapolation"]["enabled"]:
        for cutoff in config["extrapolation"]["cutoffs"]:
            tasks.append(
                ("extrapolation", surrogate_name, str(cutoff), surrogate_class, config)
            )

    if config["sparse"]["enabled"]:
        for factor in config["sparse"]["factors"]:
            tasks.append(
                ("sparse", surrogate_name, str(factor), surrogate_class, config)
            )

    if config["UQ"]["enabled"]:
        n_models = config["UQ"]["n_models"]
        for i in range(n_models - 1):
            tasks.append(("UQ", surrogate_name, str(i + 1), surrogate_class, config))

    return tasks


def parallel_training(tasks, device_list):
    """
    Execute the training tasks in parallel across multiple devices.

    Args:
        tasks (list): A list of tasks to execute in parallel.
        device_list (list): A list of devices to use for parallel training.
    """

    with ProcessPoolExecutor(max_workers=len(device_list)) as executor:
        future_to_task = {
            executor.submit(
                train_and_save_model, *task, device_list[i % len(device_list)]
            ): task
            for i, task in enumerate(tasks)
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            print_task = task[:3]
            try:
                future.result()
                print(f"Completed training for task: {print_task}")
            except Exception as e:
                print(f"Exception for task {print_task}: {e}")


def main():
    tasks = []
    device_list = config["devices"]
    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            surrogate_class = surrogate_classes[surrogate_name]
            tasks += train_surrogate(config, surrogate_class, surrogate_name)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")

    if len(device_list) > 1:
        parallel_training(tasks, device_list)
    else:
        for task in tasks:
            train_and_save_model(*task, device_list[0])


if __name__ == "__main__":
    main()