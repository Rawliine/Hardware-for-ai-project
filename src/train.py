"""
train.py
Full implementation of automated hyperparameter testing for CNN and SNN experiments
on the DVSGesture dataset. This version now includes a "super_deep" (ResNet-18) architecture
for both CNN and SNN models and an improved STDP update mechanism.
"""

import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from src.data import prepare_input, get_dataloaders
from src.models import CNNModel, SNNModel
from src.evaluate import (
    measure_energy_nvml,
    evaluate_model,
    estimate_energy_cnn,
    estimate_energy_snn
)

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

##############################
# Helper functions
##############################

def add_noise(data, noise_level):
    """Adds Gaussian noise to the data (noise_level as a fraction)."""
    if noise_level > 0:
        noise = torch.randn_like(data) * noise_level * data.std()
        return data + noise
    return data

def get_optimizer(model, optimizer_name, lr):
    """Returns an optimizer given its name."""
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

##############################
# Training functions
##############################

def train_model(model, train_loader, criterion, optimizer, epochs, noise_level):
    """Standard training loop with backpropagation (for CNN and BP-STDP SNN)."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data = prepare_input(data)
            data = add_noise(data, noise_level)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += data.size(0)
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")
    return

def stdp_update_improved(layer, lif_layer, stdp_lr, A_plus=0.01, A_minus=0.012, alpha=0.9):
    """
    Improved STDP update using the full spike train recorded in the LIF layer.
    An exponential moving average (trace) is computed over time,
    and the weight update is based on the squared correlation of this trace.
    """
    if not hasattr(lif_layer, 'spike_train') or len(lif_layer.spike_train) == 0:
        return
    spike_tensor = torch.stack(lif_layer.spike_train, dim=0)  # shape: [T, batch, ...]
    trace = spike_tensor[0]
    for t in range(1, spike_tensor.size(0)):
        trace = alpha * trace + (1 - alpha) * spike_tensor[t]
    correlation = torch.mean(trace * trace)
    delta_w = stdp_lr * (A_plus * correlation - A_minus * (1 - correlation))
    with torch.no_grad():
        layer.weight += delta_w

def train_model_snn_stdp(model, train_loader, epochs, stdp_lr, spike_rate_target, noise_level):
    """
    Custom training loop for SNN using an improved STDP update.
    For each (synaptic_layer, LIF_layer) pair, the recorded spike train is used
    to compute an exponential trace and update the weights.
    """
    model.train()
    for epoch in range(epochs):
        total_samples = 0
        start_time = time.time()
        for data, labels in train_loader:
            data = data.to(DEVICE)
            data = prepare_input(data)
            data = add_noise(data, noise_level)
            # Reset spike trains for all LIF layers in STDP pairs
            for (_, lif_layer) in model.stdp_pairs:
                lif_layer.spike_train = []
            _ = model(data)
            # Update each synaptic layer based on the corresponding LIF layer's spike train
            for (syn_layer, lif_layer) in model.stdp_pairs:
                stdp_update_improved(syn_layer, lif_layer, stdp_lr)
            total_samples += data.size(0)
        epoch_time = time.time() - start_time
        print(f"[STDP Epoch {epoch+1}/{epochs}] Time: {epoch_time:.2f}s")
    return

##############################
# Experiment functions
##############################

def run_experiment_cnn(params):
    """Runs one experiment for the CNN given a set of hyperparameters."""
    print("\n=== Running CNN Experiment ===")
    print("Hyperparameters:", params)
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    
    model = CNNModel(num_channels=5, num_classes=11, arch=params["arch"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, params["optimizer"], params["lr"])
    
    print("\nMeasuring energy during CNN training...")
    train_energy, _ = measure_energy_nvml(
        train_model, model, train_loader, criterion, optimizer, epochs=params["epochs"], noise_level=params["noise"]
    )
    print(f"CNN Training Energy: {train_energy:.4f} J")
    
    print("\nEvaluating CNN...")
    test_accuracy, test_latency = evaluate_model(model, test_loader)
    print(f"CNN Accuracy: {test_accuracy:.2f}%, Avg Batch Latency: {test_latency:.4f}s")
    
    cnn_macs = estimate_energy_cnn(model, input_shape=(5, 128, 128))
    print(f"Estimated CNN MACs: {cnn_macs}")
    
    result = {
        "Model": "CNN",
        "epochs": params["epochs"],
        "lr": params["lr"],
        "noise": params["noise"],
        "arch": params["arch"],
        "optimizer": params["optimizer"],
        "train_energy_J": train_energy,
        "test_accuracy_%": test_accuracy,
        "test_latency_s": test_latency,
        "estimated_MACs": cnn_macs
    }
    return result

def run_experiment_snn(params):
    """
    Runs one experiment for the SNN given a set of hyperparameters.
    Depending on the training method, either standard backprop (BP-STDP) or the custom STDP training is used.
    """
    print("\n=== Running SNN Experiment ===")
    print("Hyperparameters:", params)
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    
    model = SNNModel(num_channels=5, num_classes=11, time_window=params["time_window"], arch=params["arch"]).to(DEVICE)
    
    if params["training_method"] == "BP-STDP":
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, params["optimizer"], params["lr"])
        print("\nMeasuring energy during SNN (BP-STDP) training...")
        train_energy, _ = measure_energy_nvml(
            train_model, model, train_loader, criterion, optimizer, epochs=params["epochs"], noise_level=params["noise"]
        )
    else:  # Use improved STDP training
        stdp_lr = params["lr"]
        spike_rate_target = params["spike_rate"]  # Not directly used in the improved update, but kept for record
        print("\nMeasuring energy during SNN (STDP) training...")
        train_energy, _ = measure_energy_nvml(
            train_model_snn_stdp, model, train_loader, epochs=params["epochs"], stdp_lr=stdp_lr, spike_rate_target=spike_rate_target, noise_level=params["noise"]
        )
    
    print(f"SNN Training Energy: {train_energy:.4f} J")
    
    print("\nEvaluating SNN...")
    test_accuracy, test_latency = evaluate_model(model, test_loader)
    print(f"SNN Accuracy: {test_accuracy:.2f}%, Avg Batch Latency: {test_latency:.4f}s")
    
    snn_ops = estimate_energy_snn(model, input_shape=(5, 128, 128), time_window=params["time_window"], spike_rate=params["spike_rate"])
    print(f"Estimated SNN Spike Operations: {snn_ops}")
    
    result = {
        "Model": "SNN",
        "epochs": params["epochs"],
        "lr": params["lr"],
        "noise": params["noise"],
        "arch": params["arch"],
        "time_window": params["time_window"],
        "training_method": params["training_method"],
        "optimizer": params.get("optimizer", "N/A"),
        "spike_rate": params["spike_rate"],
        "train_energy_J": train_energy,
        "test_accuracy_%": test_accuracy,
        "test_latency_s": test_latency,
        "estimated_spike_ops": snn_ops
    }
    return result

def main_experiment():
    all_results = []
    
    # Define hyperparameter grids for CNN experiments
    cnn_epochs_list = [10, 50, 100, 200]
    lr_list = [0.001, 0.01, 0.1]
    noise_list = [0.0, 0.05, 0.1, 0.2]
    arch_list = ["simple", "deep", "super_deep"]
    optimizer_list = ["Adam", "SGD", "RMSProp"]
    
    for (epochs, lr, noise, arch, opt_name) in itertools.product(cnn_epochs_list, lr_list, noise_list, arch_list, optimizer_list):
        params = {
            "epochs": epochs,
            "lr": lr,
            "noise": noise,
            "arch": arch,
            "optimizer": opt_name
        }
        res = run_experiment_cnn(params)
        all_results.append(res)
    
    # Define hyperparameter grids for SNN experiments
    snn_epochs_list = [10, 50, 100, 200]
    time_window_list = [5, 10, 20, 50]  # timesteps
    spike_rate_list = [0.1, 0.5, 1.0]     # target spike rates (for record)
    training_method_list = ["BP-STDP", "STDP"]
    
    for (epochs, lr, noise, arch, time_window, spike_rate, training_method) in itertools.product(
            snn_epochs_list, lr_list, noise_list, arch_list, time_window_list, spike_rate_list, training_method_list):
        if training_method == "BP-STDP":
            for opt_name in optimizer_list:
                params = {
                    "epochs": epochs,
                    "lr": lr,
                    "noise": noise,
                    "arch": arch,
                    "time_window": time_window,
                    "spike_rate": spike_rate,
                    "training_method": training_method,
                    "optimizer": opt_name
                }
                res = run_experiment_snn(params)
                all_results.append(res)
        else:
            params = {
                "epochs": epochs,
                "lr": lr,
                "noise": noise,
                "arch": arch,
                "time_window": time_window,
                "spike_rate": spike_rate,
                "training_method": training_method
            }
            res = run_experiment_snn(params)
            all_results.append(res)
    
    # Save results to an Excel file
    df = pd.DataFrame(all_results)
    excel_filename = "experiment_results.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"\n===== ALL EXPERIMENTS COMPLETED. RESULTS SAVED TO {excel_filename} =====")

if __name__ == "__main__":
    main_experiment()
