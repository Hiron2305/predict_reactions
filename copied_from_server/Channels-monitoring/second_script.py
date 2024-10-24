import torch
import joblib
import logging
import os
import pandas as pd
from tqdm import tqdm

logging.basicConfig(filename='data_preparation.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

NUM_EMBEDDINGS = 10000
EMBEDDING_DIM = 768
TRAIN_TEST_SPLIT_RATIO = 0.8

embedding_layer = torch.nn.Embedding(NUM_EMBEDDINGS, EMBEDDING_DIM)

def generate_embeddings(df, text_column='Text', id_column='ID', batch_size=10000, save_dir='.'):
    unique_texts = df[[id_column, text_column]].drop_duplicates(subset=[id_column])

    tqdm.pandas(desc="Generating text indexes")
    unique_texts['Text_Index'] = unique_texts[text_column].progress_apply(
        lambda x: torch.tensor(hash(x) % NUM_EMBEDDINGS)
    )

    train_size = int(len(unique_texts) * TRAIN_TEST_SPLIT_RATIO)
    train_texts = unique_texts[:train_size]
    test_texts = unique_texts[train_size:]

    generate_and_save_batches(train_texts, 'train', batch_size, save_dir)

    generate_and_save_batches(test_texts, 'test', batch_size, save_dir)

def generate_and_save_batches(df, dataset_type, batch_size, save_dir):
    embeddings = []
    for i in tqdm(range(0, df.shape[0], batch_size), desc=f"Saving {dataset_type} embeddings by batch"):
        batch = df.iloc[i:i + batch_size]
        batch_embeddings = batch['Text_Index'].apply(lambda x: embedding_layer(x).detach().numpy()).tolist()
        embeddings.extend(batch_embeddings)

        batch_filename = os.path.join(save_dir, f"X_batch_{dataset_type}_{i // batch_size}.joblib")
        joblib.dump(batch_embeddings, batch_filename)

        logging.info(f"Saved {dataset_type} batch {i // batch_size} with {len(batch_embeddings)} embeddings to {batch_filename}")

    y_data = torch.randn(len(embeddings), 3)
    y_batch_filename = os.path.join(save_dir, f"y_batch_{dataset_type}_{i // batch_size}.joblib")
    joblib.dump(y_data.numpy(), y_batch_filename)
    logging.info(f"Saved {dataset_type} labels for batch {i // batch_size} to {y_batch_filename}")

if __name__ == '__main__':
    data_path = 'cleaned_data.csv'
    save_directory = '.'

    if not os.path.exists(data_path):
        logging.error(f"{data_path} not found. Please ensure the file exists.")
        raise FileNotFoundError(f"{data_path} not found.")

    logging.info("Reading cleaned_data.csv")
    df = pd.read_csv(data_path)

    if 'Text' not in df.columns or 'ID' not in df.columns:
        logging.error("Required columns 'Text' or 'ID' not found in DataFrame.")
        raise ValueError("Required columns 'Text' or 'ID' not found in DataFrame.")

    logging.info("Generating and saving embeddings.")
    generate_embeddings(df, save_dir=save_directory)

    logging.info("Data generation complete.")
