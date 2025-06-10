# src/02_feature_engineering.py

import pandas as pd
from tqdm import tqdm

def main():
    print("Starting feature engineering for 'Prev_passenger'...")
    
    # Load the preprocessed data
    data = pd.read_csv('../data/data_preprocessed.csv')
    
    # Ensure 'Date' is a datetime object for sorting
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Orig', 'Dest', 'Date']).reset_index(drop=True)

    # Group by route to calculate previous month's passengers
    data['Prev_passenger'] = data.groupby(['Orig', 'Dest'])['Reported.Est..Pax'].shift(1)

    # For the first month of any route, there's no previous data. We'll fill with 0.
    data['Prev_passenger'] = data['Prev_passenger'].fillna(0)
    
    # Save the data with the new feature
    output_path = '../data/data_with_features.csv'
    data.to_csv(output_path, index=False)

    print(f"Feature engineering complete. Data saved to {output_path}")
    print(data[['Orig', 'Dest', 'Date', 'Reported.Est..Pax', 'Prev_passenger']].head())

if __name__ == '__main__':
    main()