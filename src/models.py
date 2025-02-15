"""
models.py
Defines the CNN and SNN models.
For both models, an 'arch' parameter allows testing different architectures.
For SNNModel, an improved STDP mechanism is enabled via recording full spike trains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, layer, surrogate
from torchvision.models import resnet18

##############################
# Super Deep CNN: ResNet-18 Variant
##############################

class CNNResNet18(nn.Module):
    def __init__(self, num_channels=5, num_classes=11):
        super(CNNResNet18, self).__init__()
        self.model = resnet18(pretrained=False)
        # Modify first convolution to accept num_channels instead of 3
        self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the fully-connected layer to output num_classes
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

##############################
# Spiking ResNet-18 Components
##############################

class SpikingBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, surrogate_fn=None):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = layer.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate_fn)
        self.conv2 = layer.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate_fn)
        self.downsample = downsample
        self.stride = stride
        # Prepare lists for recording spike trains (for improved STDP)
        self.lif1.spike_train = []
        self.lif2.spike_train = []
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        # Record the output of the first LIF layer
        self.lif1.spike_train.append(out.clone())
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.lif2(out)
        self.lif2.spike_train.append(out.clone())
        return out

class SpikingResNet18(nn.Module):
    def __init__(self, num_channels=5, num_classes=11, time_window=4):
        super(SpikingResNet18, self).__init__()
        self.time_window = time_window
        self.surrogate_fn = surrogate.ATan()
        self.inplanes = 64
        self.conv1 = layer.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = neuron.LIFNode(surrogate_function=self.surrogate_fn)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                layer.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(SpikingBasicBlock(self.inplanes, planes, stride, downsample, surrogate_fn=self.surrogate_fn))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(SpikingBasicBlock(self.inplanes, planes, surrogate_fn=self.surrogate_fn))
        return nn.Sequential(*layers)
    def forward(self, x):
        # Reset network state
        functional.reset_net(self)
        self.lif1.spike_train = []
        outputs = []
        for t in range(self.time_window):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lif1(out)
            self.lif1.spike_train.append(out.clone())
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            outputs.append(out)
        avg_output = sum(outputs) / self.time_window
        return avg_output

##############################
# CNNModel: Chooses among "simple", "deep", and "super_deep"
##############################

class CNNModel(nn.Module):
    def __init__(self, num_channels=5, num_classes=11, arch="simple"):
        super(CNNModel, self).__init__()
        if arch == "simple":
            self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 32 * 32, 128)
            self.fc2 = nn.Linear(128, num_classes)
        elif arch == "deep":
            self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 16 * 16, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
        elif arch == "super_deep":
            self.model = CNNResNet18(num_channels=num_channels, num_classes=num_classes)
        else:
            raise ValueError("Unknown architecture type.")
    def forward(self, x):
        if hasattr(self, 'model'):
            return self.model(x)
        if hasattr(self, 'conv3'):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

##############################
# SNNModel: Chooses among "simple", "deep", and "super_deep"
##############################

class SNNModel(nn.Module):
    def __init__(self, num_channels=5, num_classes=11, time_window=4, arch="simple"):
        super(SNNModel, self).__init__()
        self.time_window = time_window
        self.surrogate_fn = surrogate.ATan()
        if arch == "simple":
            self.conv1 = layer.Conv2d(num_channels, 16, kernel_size=3, padding=1)
            self.lif1  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1)
            self.lif2  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.fc1 = layer.Linear(32 * 32 * 32, 128)
            self.lif3 = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.fc2 = layer.Linear(128, num_classes)
            # For STDP training, define pairs and use spike_train recording
            self.stdp_pairs = [(self.conv1, self.lif1), (self.conv2, self.lif2), (self.fc1, self.lif3)]
        elif arch == "deep":
            self.conv1 = layer.Conv2d(num_channels, 16, kernel_size=3, padding=1)
            self.lif1  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1)
            self.lif2  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.lif3  = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = layer.Linear(64 * 16 * 16, 256)
            self.lif4 = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.fc2 = layer.Linear(256, 128)
            self.lif5 = neuron.LIFNode(surrogate_function=self.surrogate_fn)
            self.fc3 = layer.Linear(128, num_classes)
            self.stdp_pairs = [(self.conv1, self.lif1), (self.conv2, self.lif2),
                               (self.conv3, self.lif3), (self.fc1, self.lif4), (self.fc2, self.lif5)]
        elif arch == "super_deep":
            self.model = SpikingResNet18(num_channels=num_channels, num_classes=num_classes, time_window=time_window)
            self.stdp_pairs = []  # (Optional: define STDP pairs for the spiking ResNet if desired)
        else:
            raise ValueError("Unknown architecture type.")
    def forward(self, x):
        functional.reset_net(self)
        if hasattr(self, 'model'):
            return self.model(x)
        # For simple and deep architectures, clear recorded spike trains
        for _, lif_layer in self.stdp_pairs:
            lif_layer.spike_train = []
        outputs = []
        for t in range(self.time_window):
            if hasattr(self, 'conv3'):
                # deep architecture
                out = self.conv1(x)
                out = self.lif1(out)
                self.lif1.spike_train.append(out.clone())
                out = F.max_pool2d(out, 2)
                out = self.conv2(out)
                out = self.lif2(out)
                self.lif2.spike_train.append(out.clone())
                out = F.max_pool2d(out, 2)
                out = self.conv3(out)
                out = self.lif3(out)
                self.lif3.spike_train.append(out.clone())
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = self.lif4(out)
                self.lif4.spike_train.append(out.clone())
                out = F.relu(out)
                out = self.fc2(out)
                out = self.lif5(out)
                self.lif5.spike_train.append(out.clone())
                out = F.relu(out)
                out = self.fc3(out)
            else:
                # simple architecture
                out = self.conv1(x)
                out = self.lif1(out)
                self.lif1.spike_train.append(out.clone())
                out = F.max_pool2d(out, 2)
                out = self.conv2(out)
                out = self.lif2(out)
                self.lif2.spike_train.append(out.clone())
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = self.lif3(out)
                self.lif3.spike_train.append(out.clone())
                out = F.relu(out)
                out = self.fc2(out)
            outputs.append(out)
        avg_output = sum(outputs) / self.time_window
        return avg_output
