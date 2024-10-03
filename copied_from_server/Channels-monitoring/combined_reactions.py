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
import ast
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
    from concurrent.futures import ThreadPoolExecutor
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

    reaction_vector = np.zeros(len(emoji_to_id), dtype=np.float32)
    for emoji, count in reactions_dict.items():
        reaction_idx = emoji_to_id.get(emoji, -1)
        if reaction_idx != -1 and isinstance(count, (int, float)):
            reaction_vector[reaction_idx] = count
    return reaction_vector

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

def split_data_and_save(X, y, test_size=0.2, batch_size=10000, X_prefix='X_batch', y_prefix='y_batch'):
    logging.info('Splitting data into training and testing sets...')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def save_batches(data, prefix, batch_size):
        num_batches = data.shape[0] // batch_size + 1
        filenames = []
        for i in range(num_batches):
            batch = data[i * batch_size: (i + 1) * batch_size]
            batch_filename = f"{prefix}_{i}.joblib"
            joblib.dump(batch, batch_filename)
            filenames.append(batch_filename)

            del batch
            gc.collect()

        return filenames

    X_train_files = save_batches(X_train, X_prefix + '_train', batch_size)
    X_test_files = save_batches(X_test, X_prefix + '_test', batch_size)
    y_train_files = save_batches(y_train, y_prefix + '_train', batch_size)
    y_test_files = save_batches(y_test, y_prefix + '_test', batch_size)

    logging.info('Data split and saved successfully.')
    return X_train_files, X_test_files, y_train_files, y_test_files

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

        logging.info("Vectorizing text data using DistilBERT...")
        preprocessed_df = pd.DataFrame(list(preprocessed_data.items()), columns=['ID', 'processed_text'])
        df['ID'] = df['ID'].astype(str)
        preprocessed_df['ID'] = preprocessed_df['ID'].astype(str)
        df = pd.merge(df, preprocessed_df, on='ID')
        X_text = np.vstack(df['processed_text'].values)

        logging.info("Saving combined features to disk in batches...")
        y = np.vstack(df['Reactions'].values)

        logging.info("Splitting data into train and test...")
        X_train_files, X_test_files, y_train_files, y_test_files = split_data_and_save(X_text, y, test_size=0.2, batch_size=10000)

        logging.info("Training and evaluation process can proceed from here...")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        #logging.error(traceback.format_exc())

if __name__ == '__main__':
    logging.info('Loading tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    logging.info('Loaded tokenizer')

    logging.info('Loading model...')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', torch_dtype=torch.float16).to(DEVICE)
    logging.info('Loaded model')

    main()
