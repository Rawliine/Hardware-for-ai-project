"""
evaluate.py
Defines functions to estimate energy (MACs for CNN and spike operations for SNN)
and to measure GPU energy consumption via NVML as well as model evaluation (accuracy and latency).
"""
from src.data import prepare_input
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer as snn_layer
import threading
import time
import pynvml

def estimate_energy_cnn(model, input_shape=(5, 128, 128)):
    """
    Estimates the total number of MAC operations for a CNN model by
    running a dummy input through the network and using forward hooks
    to capture the output shape of each Conv2d and Linear layer.
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, *input_shape), device=device)
    ops_dict = {}

    def conv_hook(module, inputs, output):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * in_channels
        _, _, H_out, W_out = output.shape
        ops = kernel_ops * out_channels * H_out * W_out
        ops_dict[module] = ops

    def linear_hook(module, inputs, output):
        ops = module.in_features * module.out_features
        ops_dict[module] = ops

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    total_macs = sum(ops_dict.values())
    return total_macs

def estimate_energy_snn(model, input_shape=(5, 128, 128), time_window=4, spike_rate=0.1):
    """
    Estimates the total number of spike operations for an SNN model.
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, *input_shape), device=device)
    ops_dict = {}

    def conv_hook(module, inputs, output):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * in_channels
        _, _, H_out, W_out = output.shape
        ops = kernel_ops * out_channels * H_out * W_out
        ops_dict[module] = ops

    def linear_hook(module, inputs, output):
        ops = module.in_features * module.out_features
        ops_dict[module] = ops

    hooks = []
    # Use spiking layers from spikingjelly.activation_based.layer
    for module in model.modules():
        if isinstance(module, snn_layer.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, snn_layer.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    total_ops = sum(ops_dict.values())
    estimated_spike_ops = total_ops * spike_rate * time_window
    return estimated_spike_ops

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test set.
    Returns overall accuracy and average latency per batch.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            data = torch.clamp(prepare_input(data), 0, 1)  # in case the input is not normalized
            start_time = time.time()
            outputs = model(data)
            inference_time = time.time() - start_time
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += data.size(0)
            total_time += inference_time
    accuracy = (total_correct / total_samples) * 100
    avg_latency = total_time / len(test_loader)
    return accuracy, avg_latency

def measure_energy_nvml(run_func, *args, sampling_interval=0.05, device_index=0, **kwargs):
    """
    Measures the total GPU energy consumption in Joules while run_func executes.
    Uses a sampling thread that polls NVML power usage in parallel.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def get_power_usage_watts():
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mW / 1000.0

    stop_event = threading.Event()
    power_readings = []

    def power_sampling():
        while not stop_event.is_set():
            current_time = time.time()
            power_watts = get_power_usage_watts()
            power_readings.append((current_time, power_watts))
            time.sleep(sampling_interval)

    sampling_thread = threading.Thread(target=power_sampling)
    sampling_thread.start()

    start_time = time.time()
    result = run_func(*args, **kwargs)
    end_time = time.time()

    stop_event.set()
    sampling_thread.join()

    final_time = time.time()
    final_power = get_power_usage_watts()
    power_readings.append((final_time, final_power))

    total_energy_joules = 0.0
    for i in range(1, len(power_readings)):
        t0, p0 = power_readings[i-1]
        t1, p1 = power_readings[i]
        dt = t1 - t0
        avg_power = (p0 + p1) / 2.0
        total_energy_joules += avg_power * dt

    pynvml.nvmlShutdown()
    elapsed_time = end_time - start_time
    print(f"--- NVML Energy Measurement ---")
    print(f"Execution Time: {elapsed_time:.2f}s")
    print(f"Total Energy:   {total_energy_joules:.4f} Joules")
    return total_energy_joules, result
