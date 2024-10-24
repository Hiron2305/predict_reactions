import pandas as pd
import os
import chardet
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(filename='data_preparation.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

popular_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding']


def read_csv_with_encoding(file_path):
    encoding = detect_encoding(file_path)
    if encoding:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            logging.info(f"Successfully read {file_path} with detected encoding: {encoding}")
            print(f"Successfully read {file_path} with detected encoding: {encoding}")
            print(f"Columns in {file_path}: {df.columns.tolist()}")
            return df
        except Exception as e:
            logging.warning(f"Error reading {file_path} with encoding {encoding}: {e}")
            print(f"Error reading {file_path} with encoding {encoding}: {e}")

    for enc in popular_encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
            logging.info(f"Successfully read {file_path} with popular encoding: {enc}")
            print(f"Successfully read {file_path} with popular encoding: {enc}")
            print(f"Columns in {file_path}: {df.columns.tolist()}")
            return df
        except Exception as e:
            logging.warning(f"Error reading {file_path} with popular encoding {enc}: {e}")
            print(f"Error reading {file_path} with popular encoding {enc}: {e}")

    logging.error(f"Could not read {file_path} with any encoding. Skipping.")
    print(f"Could not read {file_path} with any encoding. Skipping.")
    return None


def process_directory(directory):
    all_dataframes = []
    for root, _, files in os.walk(directory):
        channel_name = os.path.basename(root)
        messages_path = os.path.join(root, 'messages.csv')
        msg_stats_path = os.path.join(root, 'msg_stats.csv')

        if os.path.exists(messages_path) and os.path.exists(msg_stats_path):
            print(f"Reading {messages_path}")
            logging.info(f"Reading {messages_path}")
            messages_df = read_csv_with_encoding(messages_path)

            print(f"Reading {msg_stats_path}")
            logging.info(f"Reading {msg_stats_path}")
            msg_stats_df = read_csv_with_encoding(msg_stats_path)

            if messages_df is not None and msg_stats_df is not None:
                merged_df = pd.merge(messages_df, msg_stats_df, on='ID')
                merged_df['Channel'] = channel_name
                all_dataframes.append(merged_df)
            else:
                logging.warning(f"Skipping merge for {channel_name} due to missing data.")
                print(f"Skipping merge for {channel_name} due to missing data.")

    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        logging.info(f"Successfully merged {len(all_dataframes)} dataframes.")
        print(f"Successfully merged {len(all_dataframes)} dataframes.")
        return merged_df
    else:
        logging.error("No dataframes to concatenate.")
        print("No dataframes to concatenate.")
        return pd.DataFrame()


def calculate_time_differences(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        time_column = 'Date'
    elif 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        time_column = 'Time'
    else:
        logging.error("Column 'Date' or 'Time' not found in dataframe.")
        print("Column 'Date' or 'Time' not found in dataframe.")
        return df

    df['Tn_T0'] = (df[time_column] - df[time_column].min()).dt.total_seconds()

    random_timestamps = df[time_column].sample(frac=1).reset_index(drop=True)
    df['Tn_Tk'] = (df[time_column] - random_timestamps).dt.total_seconds()

    logging.info("Time differences calculated (Tn_T0 and Tn_Tk).")
    print("Time differences calculated (Tn_T0 and Tn_Tk).")

    return df


def clean_dataframe(df):
    channel_column_candidates = ['Channel', 'channel']
    text_column_candidates = ['Text', 'text']

    channel_column = next((col for col in channel_column_candidates if col in df.columns), None)
    text_column = next((col for col in text_column_candidates if col in df.columns), None)

    if not channel_column or not text_column:
        logging.error(
            f"Missing required columns in dataframe. Channel column: {channel_column}, Text column: {text_column}")
        print(f"Missing required columns in dataframe. Channel column: {channel_column}, Text column: {text_column}")
        return pd.DataFrame()

    df = df[df[channel_column] != 'channels_analyze']
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != '']
    logging.info(f"DataFrame cleaned: {df.shape[0]} rows remaining.")
    print(f"DataFrame cleaned: {df.shape[0]} rows remaining.")

    df = calculate_time_differences(df)

    return df


if __name__ == '__main__':
    directory = 'results'
    print(f"Starting to process directory: {directory}")
    logging.info(f"Starting to process directory: {directory}")
    merged_df = process_directory(directory)

    if not merged_df.empty:
        print("Merging completed. Starting to clean dataframe.")
        logging.info("Merging completed. Starting to clean dataframe.")
        cleaned_df = clean_dataframe(merged_df)
        if not cleaned_df.empty:
            cleaned_df.to_csv('cleaned_data.csv', index=False)
            logging.info("Cleaned data saved to cleaned_data.csv")
            print("Cleaned data saved to cleaned_data.csv")
        else:
            logging.error("Cleaned dataframe is empty.")
            print("Cleaned dataframe is empty.")
    else:
        logging.error("Merged dataframe is empty.")
        print("Merged dataframe is empty.")
