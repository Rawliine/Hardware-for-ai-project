import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

def estimate_energy_cnn(model, input_shape=(5, 128, 128)):
    total_macs = 0
    for layer_module in model.modules():
        if isinstance(layer_module, nn.Conv2d):
            out_channels = layer_module.out_channels
            in_channels  = layer_module.in_channels
            kernel_height, kernel_width = layer_module.kernel_size
            output_width = input_shape[1] // 2  # approximation (après pooling)
            output_height = input_shape[2] // 2
            macs = kernel_width * kernel_height * in_channels * out_channels * output_width * output_height
            total_macs += macs
        elif isinstance(layer_module, nn.Linear):
            total_macs += layer_module.in_features * layer_module.out_features
    return total_macs

def estimate_energy_snn(model, input_shape=(5, 128, 128), time_window=4):
    spike_rate = 0.1
    total_spikes = 0
    for layer_module in model.modules():
        if isinstance(layer_module, layer.Conv2d):
            out_channels = layer_module.out_channels
            out_width = input_shape[1] // 2
            out_height = input_shape[2] // 2
            total_neurons = out_channels * out_width * out_height
            total_spikes += total_neurons * spike_rate * time_window
        elif isinstance(layer_module, layer.Linear):
            total_spikes += layer_module.out_features * spike_rate * time_window
    return total_spikes
