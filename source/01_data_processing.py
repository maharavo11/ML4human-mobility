# src/01_data_preprocessing.py

import pandas as pd
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth using the Haversine formula.
    """
    R = 6371  # Earth's radius in km
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def main():
    print("Starting data preprocessing...")

    # Load initial datasets
    data = pd.read_excel('../data/update_combined.xlsx')
    iatac = pd.read_csv('../data/iata-icao.csv')
    pop_data = pd.read_excel('../data/update_pop (1).xlsx')
    gdp_data = pd.read_excel('../data/gdp_update.xlsx')
    tourism_data = pd.read_excel('../data/tourism.xlsx')

    # --- Initial Cleaning ---
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    data['Date'] = pd.to_datetime(data['Date'])
    data['month'] = data['Date'].dt.month
    
    # --- Calculate Flight Distances ---
    print("Calculating flight distances...")
    distances = []
    for _, row in data.iterrows():
        orig = row['Orig']
        dest = row['Dest']
        try:
            lat_orig = iatac.loc[iatac['iata'] == orig, 'latitude'].values[0]
            lon_orig = iatac.loc[iatac['iata'] == orig, 'longitude'].values[0]
            lat_dest = iatac.loc[iatac['iata'] == dest, 'latitude'].values[0]
            lon_dest = iatac.loc[iatac['iata'] == dest, 'longitude'].values[0]
            distances.append(haversine(lat_orig, lon_orig, lat_dest, lon_dest))
        except IndexError:
            distances.append(None) # Append None if airport code not found
            print(f"Warning: Airport code not found for Origin: {orig} or Destination: {dest}")
    data['Distance (km)'] = distances

    # --- Merge Population Data ---
    print("Merging population data...")
    pop_dict_2018 = pop_data.set_index('Country Name')['2018'].to_dict()
    pop_dict_2019 = pop_data.set_index('Country Name')['2019'].to_dict()

    data['origin_population'] = data.apply(
        lambda row: pop_dict_2018.get(row['Orig Country']) if row['month'] <= 6 else pop_dict_2019.get(row['Orig Country']),
        axis=1
    )
    data['dest_population'] = data.apply(
        lambda row: pop_dict_2018.get(row['Dest Country']) if row['month'] <= 6 else pop_dict_2019.get(row['Dest Country']),
        axis=1
    )

    # --- Merge GDP Data ---
    print("Merging GDP data...")
    gdp_dict_2018 = gdp_data.set_index('Country')['2018'].to_dict()
    gdp_dict_2019 = gdp_data.set_index('Country')['2019'].to_dict()
    
    data['Origin_gdp'] = data.apply(
        lambda row: gdp_dict_2018.get(row['Orig Country']) if row['month'] <= 6 else gdp_dict_2019.get(row['Orig Country']),
        axis=1
    )
    data['Dest_gdp'] = data.apply(
        lambda row: gdp_dict_2018.get(row['Dest Country']) if row['month'] <= 6 else gdp_dict_2019.get(row['Dest Country']),
        axis=1
    )

    # --- Merge Tourism Data ---
    print("Merging tourism data...")
    tourism_data = tourism_data.fillna(0)
    tourism_dict_2018 = tourism_data.set_index('Country Name')['2018'].to_dict()
    tourism_dict_2019 = tourism_data.set_index('Country Name')['2019'].to_dict()

    data['Origin_tourism_arrival'] = data.apply(
        lambda row: tourism_dict_2018.get(row['Orig Country']) if row['month'] <= 6 else tourism_dict_2019.get(row['Orig Country']),
        axis=1
    )
    data['Dest_tourism_arrival'] = data.apply(
        lambda row: tourism_dict_2018.get(row['Dest Country']) if row['month'] <= 6 else tourism_dict_2019.get(row['Dest Country']),
        axis=1
    )
    
    # Save the combined dataset
    output_path = '../data/data_preprocessed.csv'
    data.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Data saved to {output_path}")

if __name__ == '__main__':
    main()