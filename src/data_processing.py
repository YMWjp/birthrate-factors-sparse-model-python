import pandas as pd
import os

# Columns that are not needed and will be dropped from the DataFrame
UNUSED_COLUMNS = ['900793-annotations', 'Continent', 'Code']

def read_and_process_csv(file, common_countries):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join('../data/raw', file))
    # Filter the DataFrame to only include rows with entities in common_countries
    df = df[df['Entity'].isin(common_countries)]
    # Drop the unused columns, ignoring errors if they don't exist
    df.drop(columns=UNUSED_COLUMNS, errors='ignore', inplace=True)
    # Group by 'Entity' and 'Year', taking the first occurrence of each group
    return df.groupby(['Entity', 'Year']).agg('first').reset_index()

def process_csvs():
    # List all CSV files in the raw data directory
    csv_files = [f for f in os.listdir('../data/raw') if f.endswith('.csv')]
    common_countries = None

    # Determine the set of common countries across all CSV files
    for file in csv_files:
        df = pd.read_csv(os.path.join('../data/raw', file))
        unique_countries = set(df['Entity'].unique())
        # Initialize or update the set of common countries
        common_countries = unique_countries if common_countries is None else common_countries.intersection(unique_countries)

    # Specify the file that contains birth rate data
    birth_rate_file = 'birth-rate-vs-death-rate.csv'
    # Process the birth rate file
    birth_rate_df = read_and_process_csv(birth_rate_file, common_countries)

    # Process and merge each CSV file with the birth rate data
    for file in csv_files:
        if file == birth_rate_file:
            continue
        df = read_and_process_csv(file, common_countries)
        # Merge the current DataFrame with the birth rate DataFrame
        birth_rate_df = pd.merge(birth_rate_df, df, on=['Entity', 'Year'], how='left', suffixes=('', f'_{file}'))

    # Create the processed data directory if it doesn't exist
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')
    # Save the combined DataFrame to a CSV file
    birth_rate_df.to_csv('../data/processed/combined_data.csv', index=False)
