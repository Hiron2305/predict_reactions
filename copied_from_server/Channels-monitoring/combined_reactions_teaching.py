
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import logging
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def pad_tensors(tensors, target_length=None):
    if target_length is None:
        max_size = max(tensor.shape[0] for tensor in tensors)
    else:
        max_size = target_length
    
    padded_tensors = []
    for tensor in tensors:
        padding_size = max_size - tensor.shape[0]
        if padding_size > 0:
            padded_tensor = nn.functional.pad(tensor, (0, 0, 0, padding_size), "constant", 0)
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    
    return torch.stack(padded_tensors)


def load_batches_with_padding(file_paths, target_size=None):
    data = []
    for path in file_paths:
        logging.debug(f"Loading {path}...")
        batch = joblib.load(path)
        batch = torch.tensor(batch, dtype=torch.float32)

        if len(batch.shape) == 3 and batch.shape[0] == 1:
            batch = batch.squeeze(0)

        data.append(batch)

    return pad_tensors(data, target_size)


def train_model(X_train_files, y_train_files, X_test_files, y_test_files, input_dim, output_dim, num_epochs=10, batch_size=128, lr=0.001, fixed_seq_len=10000):
    model = LinearRegressionModel(input_dim, output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_items = 0

        for X_train_path, y_train_path in zip(X_train_files, y_train_files):
            X_train = load_batches_with_padding([X_train_path], target_size=fixed_seq_len)
            y_train = load_batches_with_padding([y_train_path], target_size=fixed_seq_len)

            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

            dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                assert outputs.shape == y_batch.shape, f"Output shape {outputs.shape} and target shape {y_batch.shape} mismatch!"
                
                loss = criterion(outputs.sum(axis=2), y_batch.sum(axis=2))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_items += y_batch.shape[1]

            del X_train, y_train
            torch.cuda.empty_cache()
        epoch_mean_loss = epoch_loss / num_items
        epoch_rmse = epoch_mean_loss ** 0.5
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_rmse:.4f}")

    logging.info("Training complete. Starting evaluation...")
    evaluate_model(model, X_test_files, y_test_files, batch_size, fixed_seq_len)


def evaluate_model(model, X_test_files, y_test_files, batch_size=128, fixed_seq_len=10000):
    model.eval()
    total_loss = 0
    num_items = 0
    criterion = nn.MSELoss()
    all_actual = []
    all_predicted = []

    with torch.no_grad():
        for X_test_path, y_test_path in zip(X_test_files, y_test_files):
            X_test = load_batches_with_padding([X_test_path], target_size=fixed_seq_len)
            y_test = load_batches_with_padding([y_test_path], target_size=fixed_seq_len)

            X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

            dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)

                assert outputs.shape == y_batch.shape, f"Output shape {outputs.shape} and target shape {y_batch.shape} mismatch!"

                loss = criterion(outputs.sum(axis=2), y_batch.sum(axis=2))
                total_loss += loss.item()
                num_items += y_batch.shape[1]

                all_actual.append(y_batch.cpu().numpy())
                all_predicted.append(outputs.cpu().numpy())

            del X_test, y_test
            torch.cuda.empty_cache()

    total_mean_loss = total_loss / num_items
    total_rmse = total_mean_loss ** 0.5
    logging.info(f"Test Loss: {total_rmse:.4f}")

    all_actual = np.concatenate(all_actual)
    all_predicted = np.concatenate(all_predicted)

    all_actual_flat = all_actual.ravel()
    all_predicted_flat = all_predicted.ravel()

    logging.info(f"Size of all_actual_flat: {all_actual_flat.shape}")
    logging.info(f"Size of all_predicted_flat: {all_predicted_flat.shape}")

    if all_actual_flat.shape == all_predicted_flat.shape:
        plot_separate_scatter(all_actual_flat, all_predicted_flat)
    else:
        logging.error(f"Size mismatch: actual size {all_actual_flat.shape}, predicted size {all_predicted_flat.shape}")


import matplotlib.pyplot as plt

def plot_separate_scatter(actual, predicted):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(range(len(actual)), actual, alpha=0.5, color='blue')
    plt.xlabel("Index")
    plt.ylabel("Actual Reactions")
    plt.title("Actual Reactions per Post")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(predicted)), predicted, alpha=0.5, color='green')
    plt.xlabel("Index")
    plt.ylabel("Predicted Reactions")
    plt.title("Predicted Reactions per Post")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    try:
        X_train_files = [f for f in os.listdir() if f.startswith('X_batch_train')]
        y_train_files = [f for f in os.listdir() if f.startswith('y_batch_train')]
        X_test_files = [f for f in os.listdir() if f.startswith('X_batch_test')]
        y_test_files = [f for f in os.listdir() if f.startswith('y_batch_test')]

        if not (X_train_files and y_train_files and X_test_files and y_test_files):
            raise FileNotFoundError("Не удалось найти файлы для обучения и тестирования.")

        input_dim = joblib.load(X_train_files[0]).shape[1]
        output_dim = joblib.load(y_train_files[0]).shape[1]

        train_model(X_train_files, y_train_files, X_test_files, y_test_files, input_dim, output_dim, num_epochs=5, batch_size=128, lr=0.001)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
