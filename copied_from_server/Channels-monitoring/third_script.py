import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import logging
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
class MultiOutputRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiOutputRegressionModel, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim * 3)

    def forward(self, x):
        return self.linear(x).view(-1, 10000, 3)



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

        logging.info(f"Loaded batch from {path}, shape: {batch.shape}")

        data.append(batch)

    return pad_tensors(data, target_size)



def train_model(X_train_files, y_train_files, X_test_files, y_test_files, input_dim, output_dim, num_epochs=10,
                batch_size=128, lr=0.001, fixed_seq_len=10000):
    model = MultiOutputRegressionModel(input_dim, output_dim).to(DEVICE)
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

            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

            dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)

                logging.info(f"Train: y_batch shape: {y_batch.shape}, outputs shape: {outputs.shape}")

                assert outputs.shape == y_batch.shape, f"Output shape {outputs.shape} and target shape {y_batch.shape} mismatch!"

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_items += y_batch.shape[1]

            del X_train, y_train
            torch.cuda.empty_cache()
        epoch_mean_loss = epoch_loss / num_items
        epoch_rmse = epoch_mean_loss ** 0.5
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_rmse:.4f}")

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

                logging.info(f"Test: y_batch shape: {y_batch.shape}, outputs shape: {outputs.shape}")

                assert outputs.shape == y_batch.shape, f"Output shape {outputs.shape} and target shape {y_batch.shape} mismatch!"

                loss = criterion(outputs, y_batch)
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

    if all_actual.shape == all_predicted.shape:
        plot_separate_scatter(all_actual.flatten(), all_predicted.flatten())
    else:
        logging.error(f"Size mismatch: actual size {all_actual.shape}, predicted size {all_predicted.shape}")


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

        import os
        logging.info(f"Current directory: {os.getcwd()}")
        logging.info(f"Files in directory: {os.listdir()}")

        X_train_files = [f for f in os.listdir() if 'X_batch_train' in f]
        y_train_files = [f for f in os.listdir() if 'y_batch_train' in f]
        X_test_files = [f for f in os.listdir() if 'X_batch_test' in f]
        y_test_files = [f for f in os.listdir() if 'y_batch_test' in f]



        if not (X_train_files and y_train_files and X_test_files and y_test_files):
            raise FileNotFoundError("Не удалось найти файлы для обучения и тестирования.")

        X_train_sample = joblib.load(X_train_files[0])
        X_train_sample = torch.tensor(X_train_sample, dtype=torch.float32)
        input_dim = X_train_sample.shape[1]

        y_train_sample = joblib.load(y_train_files[0])
        y_train_sample = torch.tensor(y_train_sample, dtype=torch.float32)
        output_dim = y_train_sample.shape[1] // 3

        train_model(X_train_files, y_train_files, X_test_files, y_test_files, input_dim, output_dim, num_epochs=5,
                    batch_size=128, lr=0.001)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
