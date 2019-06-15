import torch

from torch.utils.data import DataLoader, TensorDataset

from data.anime_face.tasks import image_data_file_name
from gans.util import torch_load


def anime_face_data_loader(size: int, batch_size: int, device: torch.device) -> DataLoader:
    file_name = image_data_file_name(size)
    data = torch_load(file_name).to(device)
    return DataLoader(TensorDataset(data),
                      batch_size=batch_size,
                      drop_last=True,
                      shuffle=True)