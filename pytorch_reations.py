import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
import logging
import joblib
import os
import json
import gc
import time
import ast
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import psutil


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_THREADS = min(24, psutil.cpu_count(logical=True))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_system_stats():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    logging.info(f"CPU usage: {psutil.cpu_percent()}%")

def tokens_to_device(r):
    return {
        'input_ids': r['input_ids'].to(DEVICE),
        'attention_mask': r['attention_mask'].to(DEVICE)
    }

def preprocess_text_bert(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = tokens_to_device(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

def preprocess_texts_bert(texts, ids, num_threads=NUM_THREADS):
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(preprocess_text_bert, text): msg_id for text, msg_id in zip(texts, ids)}
        for future in futures:
            msg_id = futures[future]
            results[msg_id] = future.result()
    return results

def save_preprocessed_texts(data, filename='preprocessed_texts_bert.joblib'):
    joblib.dump(data, filename)
    logging.info(f"Preprocessed texts saved to {filename}")

def load_preprocessed_texts(filename='preprocessed_texts_bert.joblib'):
    return joblib.load(filename)

def convert_reactions_to_ids(reactions, emoji_to_id):
    if isinstance(reactions, str):
        try:
            reactions_dict = ast.literal_eval(reactions)
        except ValueError:
            reactions_dict = {}
    else:
        reactions_dict = reactions if isinstance(reactions, dict) else {}

    return {emoji_to_id.get(emoji, -1): count for emoji, count in reactions_dict.items() if isinstance(count, (int, float))}


def save_emoji_mapping(emoji_to_id, filename='emoji_mapping.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(emoji_to_id, f, ensure_ascii=False, indent=4)

def convert_reactions_to_numeric(reactions):
    return np.array([sum(reactions.values())])

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
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
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            predictions = model(X_test_tensor)
            all_predictions.append(predictions.cpu().numpy())

            end_idx = current_idx + X_test_tensor.shape[0]
            y_batch = y_test.iloc[current_idx:end_idx].values

            if len(y_batch) != X_test_tensor.shape[0]:
                y_batch = pad_batch(y_batch.reshape(-1, len(emoji_to_id)), X_test_tensor.shape[0])

            all_targets.append(y_batch)
            current_idx = end_idx

        all_predictions = np.vstack(all_predictions)
        all_targets = np.concatenate(all_targets)

        all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.float32).to(DEVICE)
        all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32).to(DEVICE)

        mse = nn.MSELoss()(all_predictions_tensor, all_targets_tensor)
        return mse.item()

def train_regression_model_batch(model, X_train_files, y_train_files, emoji_to_id, batch_size=1024, epochs=15, learning_rate=0.01, save_path='model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_filename, y_filename in zip(X_train_files, y_train_files):
            X_batch = joblib.load(X_filename)
            y_batch = joblib.load(y_filename)

            y_batch_processed = []
            for val in y_batch:
                if isinstance(val, dict):
                    reaction_vector = np.zeros(len(emoji_to_id), dtype=np.float32)
                    for emoji, count in val.items():
                        reaction_idx = emoji_to_id.get(emoji, -1)
                        if reaction_idx != -1 and isinstance(count, (int, float)):
                            reaction_vector[reaction_idx] = count
                    y_batch_processed.append(reaction_vector)
                elif isinstance(val, np.ndarray):  
                    y_batch_processed.append(val)
                elif isinstance(val, (int, float)):
                    y_batch_processed.append(np.full(len(emoji_to_id), val, dtype=np.float32))
                else:
                    logging.error(f"Unexpected value in y_batch: {val}. Filling with zeros.")
                    y_batch_processed.append(np.zeros(len(emoji_to_id), dtype=np.float32))

            y_batch_processed = np.array(y_batch_processed, dtype=np.float32)

            if len(y_batch_processed) != X_batch.shape[0]:
                X_batch = pad_batch(X_batch, batch_size)
                y_batch_processed = pad_batch(y_batch_processed, batch_size)

            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
            y_batch_tensor = torch.tensor(y_batch_processed, dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch_tensor)
            loss = criterion(outputs, y_batch_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            del X_batch, X_batch_tensor, y_batch_tensor
            gc.collect()

        logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
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

def save_labels_to_disk(y, batch_size=500, filename_prefix='labels_batch'):
    num_batches = y.shape[0] // batch_size + 1
    filenames = []
    for i in range(num_batches):
        y_batch = y[i * batch_size: (i + 1) * batch_size]
        
        batch_filename = f"{filename_prefix}_{i}.joblib"
        joblib.dump(y_batch, batch_filename)
        filenames.append(batch_filename)

    return filenames

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

def cleanup_combined_features_files(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            logging.warning(f"File {filename} does not exist and could not be deleted.")

def calculate_reaction_change(df):
    df['Reaction_Change'] = None

    for post_id, group in df.groupby('ID'):
        group = group.sort_values('Date')
        initial_reactions = group.iloc[0]['Reactions']
        final_reactions = group.iloc[-1]['Reactions']

        reaction_change = {emoji: final_reactions.get(emoji, 0) - initial_reactions.get(emoji, 0) for emoji in set(initial_reactions) | set(final_reactions)}

        for idx in group.index:
            df.at[idx, 'Reaction_Change'] = reaction_change

    return df['Reaction_Change']



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

        df['Reaction_Change'] = calculate_reaction_change(df)

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

        logging.info("Vectorizing text data using DistilBERT...")
        preprocessed_df = pd.DataFrame(list(preprocessed_data.items()), columns=['ID', 'processed_text'])
        df['ID'] = df['ID'].astype(str)
        preprocessed_df['ID'] = preprocessed_df['ID'].astype(str)
        df = pd.merge(df, preprocessed_df, on='ID')
        X_text = np.vstack(df['processed_text'].values)

        logging.info("Generating time vectors in batches...")
        X_time = np.vstack([batch for batch in generate_time_vectors_batch(df)])

        logging.info("Saving combined features to disk in batches...")
        combined_features_files = save_combined_features_to_disk(X_text, X_time, batch_size=10000)

        df['Reaction_Change'] = df['Reaction_Change'].apply(lambda x: convert_reactions_to_ids(x, emoji_to_id))
        y = np.vstack(df['Reaction_Change'].values)

        logging.info("Saving labels to disk in batches...")
        y_files = save_labels_to_disk(y, batch_size=10000)

        logging.info("Splitting data...")
        X_train_files, X_test_files, y_train_files, y_test_files = train_test_split(
            combined_features_files, y_files, test_size=0.2, random_state=42
        )
        
        logging.info("Training model in batches...")
        model = LinearRegressionModel(X_text.shape[1] + X_time.shape[1], y.shape[1]).to(DEVICE)
        train_regression_model_batch(model, X_train_files, y_train_files, emoji_to_id)
        
        logging.info("Evaluating model...")
        mse = evaluate_model(model, X_test, y_test)
        logging.info(f"Mean Squared Error: {mse}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'combined_features_files' in locals():
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

