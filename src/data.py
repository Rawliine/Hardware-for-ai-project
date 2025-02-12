import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import tonic
import tonic.transforms as transforms

# Configuration de la graine pour la reproductibilité
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Définir la taille du capteur et le transform
SENSOR_SIZE = (128, 128, 2)
transform = transforms.ToVoxelGrid(sensor_size=SENSOR_SIZE, n_time_bins=5)

class Float32Wrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        event_tensor, label = self.dataset[idx]
        event_tensor = torch.tensor(event_tensor, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return event_tensor, label

def get_dataloaders(batch_size=32):
    """Charge les datasets et retourne les DataLoaders d'entraînement et de test."""
    train_dataset = Float32Wrapper(tonic.datasets.DVSGesture(save_to="./data", train=True, transform=transform))
    test_dataset  = Float32Wrapper(tonic.datasets.DVSGesture(save_to="./data", train=False, transform=transform))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def prepare_input(data):
    """
    Si data est un tenseur 5D (batch, time_bins, sensor_channels, height, width),
    on fusionne les dimensions du temps et des capteurs.
    Si data est déjà 4D, on le renvoie tel quel.
    """
    if data.ndim == 5:
        data = data.view(data.shape[0], -1, data.shape[3], data.shape[4])
    return data
