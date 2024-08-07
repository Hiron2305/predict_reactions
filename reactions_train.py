import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import psutil
import traceback
import sys
import time
import ast

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    NUM_THREADS = 1  # it's better to use 1 CPU core when GPU is used
else:
    DEVICE = "cpu"
    NUM_THREADS = 100

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
                # Save partial results every 1000 processed texts
                if i % 1000 == 0 and i != 0:
                    partial_filename = f'partial_preprocessed_texts_bert_{i}.pkl'
                    save_partial_preprocessed_texts(results, partial_filename)
            except Exception as e:
                logging.error(f"Error processing text: {e}")
    return results

def save_preprocessed_texts(data, filename='preprocessed_texts_bert.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Preprocessed texts saved to {filename}")

def load_preprocessed_texts(filename='preprocessed_texts_bert.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    logging.info(f"Preprocessed texts loaded from {filename}")
    return data

def save_partial_preprocessed_texts(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
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

def convert_reactions_to_numeric(reactions):
    return sum(reactions.values())

def main():
    try:
        logging.info("Reading data...")
        df = pd.read_csv('cleaned_data.csv')
        log_system_stats()

        df['Text'].fillna('', inplace=True)

        # Step 1: Convert emoji reactions to numeric IDs
        all_emojis = set()
        for reactions in df['Reactions']:
            reactions_dict = ast.literal_eval(reactions)
            all_emojis.update(reactions_dict.keys())

        emoji_to_id = {emoji: idx for idx, emoji in enumerate(all_emojis)}
        df['Reactions'] = df['Reactions'].apply(lambda x: convert_reactions_to_ids(x, emoji_to_id))

        preprocessed_filename = 'preprocessed_texts_bert.pkl'
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
        # Converting preprocessed_data to DataFrame with 'ID' and 'processed_text' columns
        preprocessed_df = pd.DataFrame(list(preprocessed_data.items()), columns=['ID', 'processed_text'])
        
        # Ensure 'ID' column in both dataframes are of the same type
        df['ID'] = df['ID'].astype(str)
        preprocessed_df['ID'] = preprocessed_df['ID'].astype(str)
        
        df = df.merge(preprocessed_df, on='ID')
        
        X_text = np.vstack(df['processed_text'].values)
        log_system_stats()

        logging.info("Generating time vectors...")
        df['Date'] = pd.to_datetime(df['Date'])
        df['hour'] = df['Date'].dt.hour
        df['day'] = df['Date'].dt.day
        df['week'] = df['Date'].dt.isocalendar().week

        X_time = df[['hour', 'day', 'week']].values

        X = np.hstack((X_text, X_time))
        log_system_stats()

        # Convert reactions to numeric values
        df['Reactions'] = df['Reactions'].apply(convert_reactions_to_numeric)

        y_hour = df['Reactions']
        y_day = df['Reactions']
        y_week = df['Reactions']

        logging.info("Checking if 'Reactions' column is numeric...")
        check_numeric_column(df, 'Reactions')

        logging.info("Splitting data...")
        X_train, X_test, y_train_hour, y_test_hour = train_test_split(X, y_hour, test_size=0.2, random_state=42)
        _, _, y_train_day, y_test_day = train_test_split(X, y_day, test_size=0.2, random_state=42)
        _, _, y_train_week, y_test_week = train_test_split(X, y_week, test_size=0.2, random_state=42)
        log_system_stats()

        logging.info("Training model for hourly prediction...")
        model_hour = LinearRegression()
        model_hour.fit(X_train, y_train_hour)
        log_system_stats()

        logging.info("Training model for daily prediction...")
        model_day = LinearRegression()
        model_day.fit(X_train, y_train_day)
        log_system_stats()

        logging.info("Training model for weekly prediction...")
        model_week = LinearRegression()
        model_week.fit(X_train, y_train_week)
        log_system_stats()

        logging.info("Evaluating model for hourly prediction...")
        y_pred_hour = model_hour.predict(X_test)
        mse_hour = mean_squared_error(y_test_hour, y_pred_hour)
        logging.info(f"Mean Squared Error (hourly): {mse_hour}")

        logging.info("Evaluating model for daily prediction...")
        y_pred_day = model_day.predict(X_test)
        mse_day = mean_squared_error(y_test_day, y_pred_day)
        logging.info(f"Mean Squared Error (daily): {mse_day}")

        logging.info("Evaluating model for weekly prediction...")
        y_pred_week = model_week.predict(X_test)
        mse_week = mean_squared_error(y_test_week, y_pred_week)
        logging.info(f"Mean Squared Error (weekly): {mse_week}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

logging.basicConfig(level=logging.INFO)

logging.info('loading tokenizer')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
logging.info('loaded tokenizer')

logging.info('loading model')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', torch_dtype=torch.float16).to(DEVICE)
logging.info('loaded model')

main()
