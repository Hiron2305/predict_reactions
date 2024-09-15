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
        data.append(torch.tensor(batch, dtype=torch.float32))
    
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
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_items += len(y_batch)

            del X_train, y_train
            torch.cuda.empty_cache()
        epoch_mean_loss = epoch_loss / num_items
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_mean_loss:.4f}")

    logging.info("Training complete. Starting evaluation...")
    evaluate_model(model, X_test_files, y_test_files, batch_size, fixed_seq_len)

def evaluate_model(model, X_test_files, y_test_files, batch_size=128, fixed_seq_len=10000):
    model.eval()
    total_loss = 0
    num_items = 0
    criterion = nn.MSELoss()

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
                
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                num_items += len(y_batch)

            del X_test, y_test
            torch.cuda.empty_cache()

    total_mean_loss = total_loss / num_items
    logging.info(f"Test Loss: {total_mean_loss:.4f}")



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

        train_model(X_train_files, y_train_files, X_test_files, y_test_files, input_dim, output_dim, num_epochs=10, batch_size=128, lr=0.001)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
