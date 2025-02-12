import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.data import prepare_input, get_dataloaders
from src.models import CNNModel, SNNModel

# Définir la configuration globale
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 2  # Pour démonstration, utiliser un petit nombre d'époques
LEARNING_RATE = 1e-3

def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data = prepare_input(data)
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

def evaluate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data = prepare_input(data)
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

def main_experiment():
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # Expérience CNN
    print("\n=== CNN EXPERIMENT ===")
    cnn_model = CNNModel(num_channels=5, num_classes=11).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    print("\nTraining CNN...")
    train_model(cnn_model, train_loader, criterion, optimizer_cnn, epochs=EPOCHS)
    print("\nEvaluating CNN...")
    cnn_accuracy, cnn_latency = evaluate_model(cnn_model, test_loader)
    print(f"CNN Accuracy: {cnn_accuracy:.2f}%, Avg Batch Latency: {cnn_latency:.4f}s")

    # Estimation approximative des MACs
    from src.evaluate import estimate_energy_cnn
    cnn_macs = estimate_energy_cnn(cnn_model, input_shape=(5, 128, 128))
    print(f"Estimated CNN MACs: {cnn_macs}\n")

    # Expérience SNN
    print("\n=== SNN EXPERIMENT ===")
    snn_model = SNNModel(num_channels=5, num_classes=11, time_window=4).to(DEVICE)
    criterion_snn = nn.CrossEntropyLoss()
    optimizer_snn = optim.Adam(snn_model.parameters(), lr=LEARNING_RATE)
    print("\nTraining SNN...")
    train_model(snn_model, train_loader, criterion_snn, optimizer_snn, epochs=EPOCHS)
    print("\nEvaluating SNN...")
    snn_accuracy, snn_latency = evaluate_model(snn_model, test_loader)
    print(f"SNN Accuracy: {snn_accuracy:.2f}%, Avg Batch Latency: {snn_latency:.4f}s")

    from src.evaluate import estimate_energy_snn
    snn_ops = estimate_energy_snn(snn_model, input_shape=(5, 128, 128), time_window=4)
    print(f"Estimated SNN Spike Operations: {snn_ops}\n")

    print("===== SUMMARY =====")
    print(f"CNN -> Accuracy: {cnn_accuracy:.2f}%, Latency: {cnn_latency:.4f}s, MACs: {cnn_macs}")
    print(f"SNN -> Accuracy: {snn_accuracy:.2f}%, Latency: {snn_latency:.4f}s, Spikes: {snn_ops}")

if __name__ == "__main__":
    main_experiment()
