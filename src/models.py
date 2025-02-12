import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, layer, surrogate

class CNNModel(nn.Module):
    def __init__(self, num_channels=5, num_classes=11):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SNNModel(nn.Module):
    def __init__(self, num_channels=5, num_classes=11, time_window=4):
        super(SNNModel, self).__init__()
        self.time_window = time_window
        self.surrogate_fn = surrogate.ATan()
        self.conv1 = layer.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.lif1  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
        self.fc1 = layer.Linear(32 * 32 * 32, 128)
        self.lif3 = neuron.LIFNode(surrogate_function=self.surrogate_fn)
        self.fc2 = layer.Linear(128, num_classes)

    def forward(self, x):
        functional.reset_net(self)
        outputs = []
        for t in range(self.time_window):
            out = self.conv1(x)
            out = self.lif1(out)
            out = F.max_pool2d(out, 2)
            out = self.conv2(out)
            out = self.lif2(out)
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.lif3(out)
            out = self.fc2(out)
            outputs.append(out)
        return sum(outputs) / self.time_window
