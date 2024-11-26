import pandas as pd
import os

# Columns that are not needed and will be dropped from the DataFrame
UNUSED_COLUMNS = ['900793-annotations', 'Continent', 'Code']

def read_and_process_csv(file, common_countries):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join('../data/raw', file))
    # Filter the DataFrame to only include rows with entities in common_countries
    df = df[df['Entity'].isin(common_countries)]
    # Filter the DataFrame to only include rows with Year >= 1990
    df = df[df['Year'] >= 1990]
    # Drop the unused columns, ignoring errors if they don't exist
    df.drop(columns=UNUSED_COLUMNS, errors='ignore', inplace=True)
    # Group by 'Entity' and 'Year', taking the first occurrence of each group
    return df.groupby(['Entity', 'Year']).agg('first').reset_index()

def process_csvs():
    # List all CSV files in the raw data directory
    csv_files = [f for f in os.listdir('../data/raw') if f.endswith('.csv')]

    # OECD countries
    oecd_countries = {'Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 
                    'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 
                    'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Latvia', 'Lithuania', 'Luxembourg', 
                    'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovakia', 
                    'Slovenia', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 
                    'United States'}

    # Specify the file that contains birth rate data
    birth_rate_file = 'birth-rate-vs-death-rate.csv'
    # Process the birth rate file
    birth_rate_df = read_and_process_csv(birth_rate_file, oecd_countries)

    # Process and merge each CSV file with the birth rate data
    for file in csv_files:
        if file == birth_rate_file:
            continue
        print(f"Processing file: {file}")
        try:
            df = read_and_process_csv(file, oecd_countries)
            birth_rate_df = pd.merge(birth_rate_df, df, on=['Entity', 'Year'], how='left', suffixes=('', f'_{file}'))
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # Create the processed data directory if it doesn't exist
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')
    # Save the combined DataFrame to a CSV file
    birth_rate_df.to_csv('../data/processed/combined_data.csv', index=False)
