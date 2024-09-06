import random
import string

import pytest
import torch

from surrogates.surrogate_classes import surrogate_classes

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CHEMICALS = 10
N_TIMESTEPS = 50

BATCH_SIZE = 1


@pytest.fixture(params=surrogate_classes)
def instance(request):
    return request.param(DEVICE, N_CHEMICALS, N_TIMESTEPS)


@pytest.fixture
def dataloaders(instance):
    data_train = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    data_test = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    data_val = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    timesteps = torch.rand(N_TIMESTEPS, dtype=torch.float64)
    shuffle = True
    dataloader_train, dataloader_test, dataloader_val = instance.prepare_data(
        data_train, data_test, data_val, timesteps, BATCH_SIZE, shuffle
    )
    return dataloader_train, dataloader_test, dataloader_val

def test_init(instance):
    assert instance.device == DEVICE, f"device is wrong: {instance.device} != {DEVICE}"
    assert instance.n_chemicals == N_CHEMICALS, f"n_chemicals is wrong: {instance.n_chemicals} != {N_CHEMICALS}"
    assert instance.n_timesteps == N_TIMESTEPS, f"n_timesteps is wrong: {instance.n_timesteps} != {N_TIMESTEPS}"
    assert instance.config is not None, "config is None"

def test_dataloader(instance, dataloaders):
    dataloader_train, dataloader_test, dataloader_val = dataloaders
    
    assert isinstance(dataloader_train, torch.utils.data.DataLoader), "dataloader_train is not a DataLoader"
    assert isinstance(dataloader_test, torch.utils.data.DataLoader), "dataloader_test is not a DataLoader"
    assert isinstance(dataloader_val, torch.utils.data.DataLoader), "dataloader_val is not a DataLoader"

    assert dataloader_train.batch_size == BATCH_SIZE, f"dataloader_train has wrong batch size: {dataloader_train.batch_size} != {BATCH_SIZE}"
    assert dataloader_test.batch_size == BATCH_SIZE, f"dataloader_test has wrong batch size: {dataloader_test.batch_size} != {BATCH_SIZE}"
    assert dataloader_val.batch_size == BATCH_SIZE, f"dataloader_val has wrong batch size: {dataloader_val.batch_size} != {BATCH_SIZE}"


def test_predict(instance, dataloaders):
    dataloader_train, _, _ = dataloaders
    predictions, targets = instance.predict(dataloader_train)

    assert predictions.shape == torch.Size([3, N_TIMESTEPS, N_CHEMICALS]), f"predictions has wrong shape: {predictions.shape} != [3, {N_TIMESTEPS}, {N_CHEMICALS}]"
    assert targets.shape == torch.Size([3, N_TIMESTEPS, N_CHEMICALS]), f"targets has wrong shape: {targets.shape} != [3, {N_TIMESTEPS}, {N_CHEMICALS}]"


def test_fit(instance, dataloaders):
    dataloader_train, dataloader_test, _ = dataloaders
    instance.fit(dataloader_train, dataloader_test, epochs=2)

    assert instance.train_loss.shape == torch.Size([2]), f"train_loss has wrong shape: {instance.train_loss.shape} != [2]"
    assert instance.test_loss.shape == torch.Size([2]), f"test_loss has wrong shape: {instance.test_loss.shape} != [2]"
    assert instance.MAE.shape == torch.Size([2]), f"MAE has wrong shape: {instance.MAE.shape} != [2]"


def test_save_load(instance, tmp_path):
    model_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    instance.save(
        model_name=model_name, base_dir=tmp_path, training_id="TestID", data_params={}
    )
    new_instance = instance.__class__(DEVICE, N_CHEMICALS, N_TIMESTEPS)
    new_instance.load(
        training_id="TestID",
        surr_name=instance.__class__.__name__,
        model_identifier=model_name,
        model_dir=tmp_path,
    )
    assert new_instance is not None, "model is None after loading"
