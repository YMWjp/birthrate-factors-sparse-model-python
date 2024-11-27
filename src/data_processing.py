import pandas as pd
import os
import json
# Constants for data processing
UNUSED_COLUMNS = ['900793-annotations', 'Continent', 'Code']
DATA_START_YEAR = 1990
RAW_DATA_DIR = '../data/raw'
PROCESSED_DATA_DIR = '../data/processed'
OUTPUT_FILENAME = 'combined_data.csv'
COUNTRIES_CONFIG_PATH = '../config/countries.json'

def load_oecd_countries():
    try:
        with open(COUNTRIES_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return set(config['oecd_countries'])
    except Exception as e:
        print(f"Error loading OECD countries configuration: {str(e)}")
        raise

def read_and_process_csv(file, common_countries):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file))
        
        # Filter and preprocess the DataFrame
        df = (df[df['Entity'].isin(common_countries)]
              .query(f'Year >= {DATA_START_YEAR}')
              .drop(columns=UNUSED_COLUMNS, errors='ignore'))
        
        return df.groupby(['Entity', 'Year']).agg('first').reset_index()
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        return None

def process_csvs():
    # List of OECD member countries
    oecd_countries = load_oecd_countries()

    try:
        # Process birth rate data first
        birth_rate_file = 'birth-rate-vs-death-rate.csv'
        birth_rate_df = read_and_process_csv(birth_rate_file, oecd_countries)
        
        if birth_rate_df is None:
            raise ValueError("Failed to load birth rate data")

        # Process and merge other CSV files
        csv_files = [
            f for f in os.listdir(RAW_DATA_DIR) 
            if f.endswith('.csv') and f != birth_rate_file
        ]

        for file in csv_files:
            print(f"Processing file: {file}")
            df = read_and_process_csv(file, oecd_countries)
            
            if df is not None:
                birth_rate_df = pd.merge(
                    birth_rate_df, 
                    df, 
                    on=['Entity', 'Year'], 
                    how='left', 
                    suffixes=('', f'_{file}')
                )

        # Create output directory and save results
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # 数値カラムのみを選択（'Year'は除外）
        numeric_columns = birth_rate_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns if col != 'Year']
        
        # 各年の平均値を計算（数値カラムのみ）
        yearly_means = birth_rate_df.groupby('Year')[numeric_columns].mean().reset_index()
        yearly_means['Entity'] = 'OECD Average'
        
        # 平均値を元のデータフレームに追加
        birth_rate_df = pd.concat([birth_rate_df, yearly_means], ignore_index=True)
        
        output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME)
        birth_rate_df.to_csv(output_path, index=False)
        print(f"Processing completed. Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Critical error during data processing: {str(e)}")