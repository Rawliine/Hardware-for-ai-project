"""
train.py
Full implementation of automated hyperparameter testing using Bayesian optimization (Optuna)
for CNN and SNN experiments on the DVSGesture dataset. This version includes a "super_deep" (ResNet-18)
architecture option for both CNN and SNN models and an improved STDP update mechanism.
It also saves interim results to Excel during the experiment and checkpoints each trained model.
"""

import time
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import optuna

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
BATCH_SIZE = 256

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
# Experiment functions (with model saving)
##############################

def run_experiment_cnn(params):
    print("\n=== Running CNN Experiment ===")
    print("Hyperparameters:", params)
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    
    model = CNNModel(num_channels=5, num_classes=11, arch=params["arch"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, params["optimizer"], params["lr"])
    
    print("\nMeasuring energy during CNN training...")
    train_energy, _ = measure_energy_nvml(train_model, model, train_loader, criterion, optimizer, epochs=params["epochs"], noise_level=params["noise"])
    print(f"CNN Training Energy: {train_energy:.4f} J")
    
    print("\nEvaluating CNN...")
    test_accuracy, test_latency = evaluate_model(model, test_loader)
    print(f"CNN Accuracy: {test_accuracy:.2f}%, Avg Batch Latency: {test_latency:.4f}s")
    
    cnn_macs = estimate_energy_cnn(model, input_shape=(5, 128, 128))
    print(f"Estimated CNN MACs: {cnn_macs}")
    
    # Save the model immediately
    model_filename = f"saved_models/cnn_{params['arch']}_{int(time.time())}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Saved CNN model to {model_filename}")
    
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
        "estimated_MACs": cnn_macs,
        "model_file": model_filename
    }
    return result

def run_experiment_snn(params):
    print("\n=== Running SNN Experiment ===")
    print("Hyperparameters:", params)
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    
    model = SNNModel(num_channels=5, num_classes=11, time_window=params["time_window"], arch=params["arch"]).to(DEVICE)
    
    if params["training_method"] == "BP-STDP":
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, params["optimizer"], params["lr"])
        print("\nMeasuring energy during SNN (BP-STDP) training...")
        train_energy, _ = measure_energy_nvml(train_model, model, train_loader, criterion, optimizer, epochs=params["epochs"], noise_level=params["noise"])
    else:  # Use improved STDP training
        stdp_lr = params["lr"]
        spike_rate_target = params["spike_rate"]
        print("\nMeasuring energy during SNN (STDP) training...")
        train_energy, _ = measure_energy_nvml(train_model_snn_stdp, model, train_loader, epochs=params["epochs"], stdp_lr=stdp_lr, spike_rate_target=spike_rate_target, noise_level=params["noise"])
    
    print(f"SNN Training Energy: {train_energy:.4f} J")
    
    print("\nEvaluating SNN...")
    test_accuracy, test_latency = evaluate_model(model, test_loader)
    print(f"SNN Accuracy: {test_accuracy:.2f}%, Avg Batch Latency: {test_latency:.4f}s")
    
    snn_ops = estimate_energy_snn(model, input_shape=(5, 128, 128), time_window=params["time_window"], spike_rate=params["spike_rate"])
    print(f"Estimated SNN Spike Operations: {snn_ops}")
    
    # Save the model immediately
    model_filename = f"saved_models/snn_{params['arch']}_{int(time.time())}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Saved SNN model to {model_filename}")
    
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
        "estimated_spike_ops": snn_ops,
        "model_file": model_filename
    }
    return result

##############################
# Optuna objective functions
##############################

def objective_cnn(trial):
    # Define hyperparameters using trial suggestions
    epochs = trial.suggest_categorical("epochs", [10, 30])
    lr = trial.suggest_categorical("lr", [0.001, 0.01, 0.1])
    noise = trial.suggest_categorical("noise", [0.0, 0.1, 0.2])
    arch = trial.suggest_categorical("arch", ["simple", "super_deep"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])
    
    params = {
        "epochs": epochs,
        "lr": lr,
        "noise": noise,
        "arch": arch,
        "optimizer": optimizer_name
    }
    
    result = run_experiment_cnn(params)
    # We want to maximize test accuracy, so return negative accuracy for minimization.
    return -result["test_accuracy_%"]

def objective_snn(trial):
    epochs = trial.suggest_categorical("epochs", [10, 30])
    lr = trial.suggest_categorical("lr", [0.001, 0.01, 0.1])
    noise = trial.suggest_categorical("noise", [0.0, 0.1, 0.2])
    arch = trial.suggest_categorical("arch", ["simple", "super_deep"])
    time_window = trial.suggest_categorical("time_window", [5, 20, 50])
    spike_rate = trial.suggest_categorical("spike_rate", [0.1, 1.0])
    training_method = trial.suggest_categorical("training_method", ["BP-STDP", "STDP"])
    
    if training_method == "BP-STDP":
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])
        params = {
            "epochs": epochs,
            "lr": lr,
            "noise": noise,
            "arch": arch,
            "time_window": time_window,
            "spike_rate": spike_rate,
            "training_method": training_method,
            "optimizer": optimizer_name
        }
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
    
    result = run_experiment_snn(params)
    return -result["test_accuracy_%"]

##############################
# Optuna callback to save interim results to Excel
##############################

def save_results_callback(study, trial):
    df = study.trials_dataframe()
    excel_filename = f"{study.study_name}_results.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"Saved interim results to {excel_filename}")

##############################
# Main function for Optuna hyperparameter search
##############################

def main_experiment():
    os.makedirs("saved_models", exist_ok=True)
    
    # Run CNN experiments using Optuna
    cnn_study = optuna.create_study(direction="minimize", study_name="cnn_study")
    cnn_study.optimize(objective_cnn, n_trials=40, callbacks=[save_results_callback])
    
    # Run SNN experiments using Optuna
    snn_study = optuna.create_study(direction="minimize", study_name="snn_study")
    snn_study.optimize(objective_snn, n_trials=40, callbacks=[save_results_callback])
    
    # Save combined results
    cnn_df = cnn_study.trials_dataframe()
    snn_df = snn_study.trials_dataframe()
    combined_df = pd.concat([cnn_df, snn_df], ignore_index=True)
    combined_excel = "combined_experiment_results.xlsx"
    combined_df.to_excel(combined_excel, index=False)
    print(f"\n===== ALL EXPERIMENTS COMPLETED. Combined results saved to {combined_excel} =====")

if __name__ == "__main__":
    main_experiment()
