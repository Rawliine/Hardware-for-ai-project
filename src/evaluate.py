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
    
    Args:
        model (nn.Module): The CNN model.
        input_shape (tuple): Input shape as (channels, height, width).
        
    Returns:
        total_macs (int): Total number of MAC operations.
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, *input_shape), device=device)
    ops_dict = {}

    def conv_hook(module, inputs, output):
        # inputs[0]: (1, in_channels, H, W)
        # output: (1, out_channels, H_out, W_out)
        in_channels = module.in_channels
        out_channels = module.out_channels
        # Each output pixel requires (kernel_height * kernel_width * in_channels) MACs
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * in_channels
        _, _, H_out, W_out = output.shape
        ops = kernel_ops * out_channels * H_out * W_out
        ops_dict[module] = ops

    def linear_hook(module, inputs, output):
        # For linear layers: in_features * out_features MACs per sample.
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
    The function uses forward hooks on spiking convolutional and linear
    layers (from spikingjelly) to obtain the maximum possible operations
    (if every neuron spiked), then scales the total by the average spike rate
    and the number of timesteps (time_window).
    
    Args:
        model (nn.Module): The SNN model.
        input_shape (tuple): Input shape as (channels, height, width).
        time_window (int): Number of timesteps the SNN is unrolled.
        spike_rate (float): Average fraction of neurons that spike per timestep.
        
    Returns:
        estimated_spike_ops (float): Estimated number of spike operations.
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, *input_shape), device=device)
    ops_dict = {}

    def conv_hook(module, inputs, output):
        # inputs[0]: (1, in_channels, H, W)
        # output: (1, out_channels, H_out, W_out)
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
        # Run a forward pass; note that many SNN models internally loop over time_window timesteps.
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    total_ops = sum(ops_dict.values())
    # Since spiking networks are event-driven, only a fraction of the neurons spike each timestep.
    estimated_spike_ops = total_ops * spike_rate * time_window
    return estimated_spike_ops



def measure_energy_nvml(run_func, *args, sampling_interval=0.05, device_index=0, **kwargs):
    """
    Measures the total GPU energy consumption in Joules while `run_func` executes.
    Uses a sampling thread that polls NVML power usage in parallel.

    :param run_func: The function you want to measure (e.g. training or inference function).
    :param args: Positional arguments to pass to `run_func`.
    :param sampling_interval: Time (in seconds) between power samples.
    :param device_index: Index of the GPU to monitor (default 0).
    :param kwargs: Keyword arguments to pass to `run_func`.
    :return: The total energy consumption in Joules, plus whatever `run_func` returns.
    """

    # --- Initialize NVML and get handle for the target GPU ---
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def get_power_usage_watts():
        """Return current power usage in Watts for the GPU."""
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)  # power in milliwatts
        return power_mW / 1000.0  # convert to Watts

    # --- Shared variables for the sampling thread ---
    stop_event = threading.Event()
    power_readings = []  # will store (timestamp, power_in_watts)

    def power_sampling():
        """Background thread that periodically samples GPU power."""
        while not stop_event.is_set():
            current_time = time.time()
            power_watts = get_power_usage_watts()
            power_readings.append((current_time, power_watts))
            time.sleep(sampling_interval)

    # --- Start sampling in the background ---
    sampling_thread = threading.Thread(target=power_sampling)
    sampling_thread.start()

    # --- Run the target function and measure execution time ---
    start_time = time.time()
    result = run_func(*args, **kwargs)  # this is where you do your training/inference
    end_time = time.time()

    # --- Stop sampling and wait for the thread to finish ---
    stop_event.set()
    sampling_thread.join()

    # --- One last reading to capture final power (optional) ---
    final_time = time.time()
    final_power = get_power_usage_watts()
    power_readings.append((final_time, final_power))

    # --- Compute total energy via trapezoidal integration ---
    # Each segment: energy = average_power * delta_time
    total_energy_joules = 0.0
    for i in range(1, len(power_readings)):
        t0, p0 = power_readings[i-1]
        t1, p1 = power_readings[i]
        dt = t1 - t0
        avg_power = (p0 + p1) / 2.0
        total_energy_joules += avg_power * dt

    # --- Cleanup NVML ---
    pynvml.nvmlShutdown()

    elapsed_time = end_time - start_time

    print(f"--- NVML Energy Measurement ---")
    print(f"Execution Time: {elapsed_time:.2f}s")
    print(f"Total Energy:   {total_energy_joules:.4f} Joules")

    return total_energy_joules, result
