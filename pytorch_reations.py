import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
from concurrent.futures import ThreadPoolExecutor
import joblib
import psutil
import traceback
import sys
import time
import ast
import gc
import os
from datetime import datetime
import json as json


if torch.cuda.is_available():
    DEVICE = "cuda:0"
    NUM_THREADS = 1
else:
    DEVICE = "cpu"
    NUM_THREADS = 24

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_system_stats():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    logging.info(f"CPU usage: {psutil.cpu_percent()}%")

def tokens_to_device(r):
    tokens_tensor = r['input_ids'].to(DEVICE)
    attention_mask = r['attention_mask'].to(DEVICE)
    output = {
        'input_ids': tokens_tensor,
        'attention_mask': attention_mask
    }
    return output

def preprocess_text_bert(text):
    logging.debug('before tokenizing')
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    logging.debug('tokenized')
    inputs = tokens_to_device(inputs)
    logging.debug('tokenized data on GPU')
    with torch.no_grad():
        outputs = model(**inputs)
        logging.debug('model evaluated')
    result = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    logging.debug('got final embedding data')
    return result

def preprocess_texts_bert(texts, ids, num_threads=4):
    start_time = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(preprocess_text_bert, text): msg_id for text, msg_id in zip(texts, ids)}
        for i, future in enumerate(futures):
            try:
                msg_id = futures[future]
                results[msg_id] = future.result()
                if len(results) % 50 == 0:
                    logging.info(f'Lines processed: {len(results)}')
                if len(results) >= 10000:
                    print(f'Time elapsed {time.time() - start_time} sec')
                if i % 1000 == 0 and i != 0:
                    partial_filename = f'partial_preprocessed_texts_bert_{i}.joblib'
                    save_partial_preprocessed_texts(results, partial_filename)
            except Exception as e:
                logging.error(f"Error processing text: {e}")
    return results

def save_preprocessed_texts(data, filename='preprocessed_texts_bert.joblib'):
    joblib.dump(data, filename)
    logging.info(f"Preprocessed texts saved to {filename}")

def load_preprocessed_texts(filename='preprocessed_texts_bert.joblib'):
    data = joblib.load(filename)
    logging.info(f"Preprocessed texts loaded from {filename}")
    return data

def save_partial_preprocessed_texts(data, filename):
    joblib.dump(data, filename)
    logging.info(f"Partial preprocessed texts saved to {filename}")

def check_numeric_column(df, column):
    try:
        df[column] = pd.to_numeric(df[column], errors='raise')
        logging.info(f"Column {column} successfully converted to numeric.")
    except ValueError as e:
        logging.error(f"Column {column} contains non-numeric values: {e}")
        logging.error(df[column].value_counts())
        raise

def convert_reactions_to_ids(reactions, emoji_to_id):
    reactions_dict = ast.literal_eval(reactions)
    reaction_ids = {emoji_to_id[emoji]: count for emoji, count in reactions_dict.items()}
    return reaction_ids

def save_emoji_mapping(emoji_to_id, filename='emoji_mapping.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(emoji_to_id, f, ensure_ascii=False, indent=4)
    logging.info(f"Emoji to ID mapping saved to {filename}")

def convert_reactions_to_numeric(reactions):
    return sum(reactions.values())

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def pad_batch(batch, target_size):
    if batch.shape[0] < target_size:
        pad_size = target_size - batch.shape[0]
        padding = np.zeros((pad_size, batch.shape[1]), dtype=batch.dtype)
        batch = np.vstack((batch, padding))
    return batch

def evaluate_model(model, X_test_files, y_test, batch_size=1024):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        current_idx = 0
        for filename in X_test_files:
            X_test = joblib.load(filename)
            logging.info(f"Processing file: {filename} with shape {X_test.shape}")

            if isinstance(X_test, np.ndarray):
                if not np.issubdtype(X_test.dtype, np.number):
                    logging.error(f"Data in {filename} contains non-numeric values.")
                    raise ValueError("Non-numeric data found in X_test")
            else:
                logging.error(f"Loaded data from {filename} is not an ndarray.")
                raise ValueError("Data is not a numpy ndarray")

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            predictions = model(X_test_tensor)
            all_predictions.append(predictions.cpu().numpy())

            end_idx = current_idx + X_test_tensor.shape[0]
            y_batch = y_test.iloc[current_idx:end_idx].values

            if len(y_batch) != X_test_tensor.shape[0]:
                y_batch = pad_batch(y_batch.reshape(-1, 1), X_test_tensor.shape[0]).flatten()
                logging.warning(f"Adjusted y_batch size to match X_test_tensor: {len(y_batch)}")

            all_targets.append(y_batch)
            current_idx = end_idx

        all_predictions = np.vstack(all_predictions)
        all_targets = np.concatenate(all_targets).reshape(-1, 1)


        if all_predictions.shape[0] != all_targets.shape[0]:
            logging.error(f"Size mismatch before MSE calculation: predictions size {all_predictions.shape[0]}, targets size {all_targets.shape[0]}")
            return None

        all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.float32).to(DEVICE)
        all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32).to(DEVICE)

        mse = nn.MSELoss()(all_predictions_tensor, all_targets_tensor)
        return mse.item()

import torch

def train_regression_model_batch(model, combined_features_files, y_train, batch_size=1024, epochs=15, learning_rate=0.01, save_path='model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss_hour = 0
        epoch_loss_day = 0
        epoch_loss_week = 0

        for i, filename in enumerate(combined_features_files):
            X_batch = joblib.load(filename)
            
            start_idx = i * batch_size
            end_idx = start_idx + X_batch.shape[0]
            y_batch = y_train[start_idx:end_idx].to_numpy()

            if len(y_batch) != X_batch.shape[0]:
                X_batch = pad_batch(X_batch, batch_size)
                y_batch = pad_batch(y_batch.reshape(-1, 1), batch_size).flatten()

            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
            y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch_tensor)
            loss = criterion(outputs, y_batch_tensor)
            loss.backward()
            optimizer.step()

            if 'hour' in filename:
                epoch_loss_hour += loss.item()
            elif 'day' in filename:
                epoch_loss_day += loss.item()
            elif 'week' in filename:
                epoch_loss_week += loss.item()

            del X_batch, X_batch_tensor, y_batch_tensor
            gc.collect()

        logging.info(f'Epoch [{epoch+1}/{epochs}]')
    
    torch.save(model.state_dict(), save_path)
    logging.info(f'Model saved to {save_path}')
    
    return model




def generate_time_vectors_batch(df, batch_size=10000):
    num_batches = len(df) // batch_size + 1
    for i in range(num_batches):
        batch_df = df.iloc[i * batch_size: (i + 1) * batch_size].copy()
        batch_df['Date'] = pd.to_datetime(batch_df['Date'])
        batch_df['hour'] = batch_df['Date'].dt.hour.astype(np.float32)
        batch_df['day'] = batch_df['Date'].dt.day.astype(np.float32)
        batch_df['week'] = batch_df['Date'].dt.isocalendar().week.astype(np.float32)
        yield batch_df[['hour', 'day', 'week']].values

def save_combined_features_to_disk(X_text, X_time, batch_size=500, filename_prefix='combined_features_batch'):
    num_batches = X_text.shape[0] // batch_size + 1
    filenames = []
    for i in range(num_batches):
        X_text_batch = X_text[i * batch_size: (i + 1) * batch_size]
        X_time_batch = X_time[i * batch_size: (i + 1) * batch_size]
        combined_features_batch = np.hstack((X_text_batch, X_time_batch))
        
        batch_filename = f"{filename_prefix}_{i}.joblib"
        joblib.dump(combined_features_batch, batch_filename)
        filenames.append(batch_filename)

        del combined_features_batch
        gc.collect()
        
    return filenames

def combine_text_and_time_features_batch(X_text, X_time, batch_size=500):
    num_batches = X_text.shape[0] // batch_size + 1
    for i in range(num_batches):
        X_text_batch = X_text[i * batch_size: (i + 1) * batch_size]
        X_time_batch = X_time[i * batch_size: (i + 1) * batch_size]
        combined_batch = np.hstack((X_text_batch, X_time_batch))
        yield combined_batch

def cleanup_combined_features_files(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            logging.warning(f"File {filename} does not exist and could not be deleted.")

def main():
    try:
        logging.info("Reading data...")
        df = pd.read_csv('cleaned_data.csv')
        log_system_stats()

        df = df.dropna(subset=['Text'])

        all_emojis = set()
        for reactions in df['Reactions']:
            reactions_dict = ast.literal_eval(reactions)
            all_emojis.update(reactions_dict.keys())

        emoji_to_id = {emoji: idx for idx, emoji in enumerate(all_emojis)}
        
        save_emoji_mapping(emoji_to_id)

        df['Reactions'] = df['Reactions'].apply(lambda x: convert_reactions_to_ids(x, emoji_to_id))

        preprocessed_filename = 'preprocessed_texts_bert.joblib'
        try:
            preprocessed_data = load_preprocessed_texts(preprocessed_filename)
        except FileNotFoundError:
            logging.info("Preprocessing text data using DistilBERT...")
            unique_texts = df.drop_duplicates(subset='ID')
            texts = unique_texts['Text'].tolist()
            ids = unique_texts['ID'].tolist()
            preprocessed_data = preprocess_texts_bert(texts, ids, num_threads=NUM_THREADS)
            save_preprocessed_texts(preprocessed_data, preprocessed_filename)

        log_system_stats()

        logging.info("Vectorizing text data using DistilBERT...")
        preprocessed_df = pd.DataFrame(list(preprocessed_data.items()), columns=['ID', 'processed_text'])
        df['ID'] = df['ID'].astype(str)
        preprocessed_df['ID'] = preprocessed_df['ID'].astype(str)
        df = pd.merge(df, preprocessed_df, on='ID')
        X_text = np.vstack(df['processed_text'].values)

        log_system_stats()

        logging.info("Generating time vectors in batches...")
        X_time = np.vstack([batch for batch in generate_time_vectors_batch(df)])
        log_system_stats()

        logging.info("Saving combined features to disk in batches...")
        combined_features_files = save_combined_features_to_disk(X_text, X_time, batch_size=10000)
        log_system_stats()

        df['Reactions'] = df['Reactions'].apply(convert_reactions_to_numeric)
        y_hour = df['Reactions']
        y_day = df['Reactions']
        y_week = df['Reactions']

        logging.info("Splitting data...")
        y_train_hour, y_test_hour = train_test_split(y_hour, test_size=0.2, random_state=42)
        y_train_day, y_test_day = train_test_split(y_day, test_size=0.2, random_state=42)
        y_train_week, y_test_week = train_test_split(y_week, test_size=0.2, random_state=42)
        log_system_stats()

        logging.info("Training model for hourly prediction in batches...")
        model_hour = LinearRegressionModel(X_text.shape[1] + X_time.shape[1]).to(DEVICE)
        train_regression_model_batch(model_hour, combined_features_files, y_train_hour)
        log_system_stats()

        logging.info("Training model for daily prediction in batches...")
        model_day = LinearRegressionModel(X_text.shape[1] + X_time.shape[1]).to(DEVICE)
        train_regression_model_batch(model_day, combined_features_files, y_train_day)
        log_system_stats()

        logging.info("Training model for weekly prediction in batches...")
        model_week = LinearRegressionModel(X_text.shape[1] + X_time.shape[1]).to(DEVICE)
        train_regression_model_batch(model_week, combined_features_files, y_train_week)
        log_system_stats()

        logging.info("Evaluating model for hourly prediction...")
        mse_hour = evaluate_model(model_hour, combined_features_files, y_test_hour)
        logging.info(f"Mean Squared Error (hourly): {mse_hour}")

        logging.info("Evaluating model for daily prediction...")
        mse_day = evaluate_model(model_day, combined_features_files, y_test_day)
        logging.info(f"Mean Squared Error (daily): {mse_day}")

        logging.info("Evaluating model for weekly prediction...")
        mse_week = evaluate_model(model_week, combined_features_files, y_test_week)
        logging.info(f"Mean Squared Error (weekly): {mse_week}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        cleanup_combined_features_files(combined_features_files)
        logging.info("Finished cleanup of temporary files.")

if __name__ == '__main__':
    logging.info('Loading tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    logging.info('Loaded tokenizer')

    logging.info('Loading model...')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', torch_dtype=torch.float16).to(DEVICE)
    logging.info('Loaded model')

    main()
