# src/05_predict_future.py

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from tensorflow.keras.models import load_model
from tqdm import tqdm

def interpolate_future_data(df, start_year, end_year, value_col, base_year_col, growth_rate_years):
    """Interpolates a given metric based on an average growth rate."""
    df_pred = df.copy()
    
    # Calculate average growth rate
    growth_rate = (
        (df_pred[value_col] - df_pred[base_year_col]) / df_pred[base_year_col]
    ) / growth_rate_years

    # Interpolate for future years
    for year in range(start_year, end_year + 1):
        years_since_base = year - start_year + 1
        df_pred[str(year)] = df_pred[value_col] * (1 + growth_rate) ** years_since_base
    
    return df_pred

def main():
    print("Starting future prediction pipeline...")

    # --- Load Models and Transformers ---
    print("Loading trained models, scalers, and encoders...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('../models/xgb_model.json')
    
    transformer_model = load_model('../models/transformer_model.keras')

    with open('../models/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open('../models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # --- Prepare Future Data Skeletons ---
    print("Preparing future data skeletons...")
    
    # Load historical data to get all unique routes
    data_hist = pd.read_csv('../data/data_with_features.csv')
    unique_routes = data_hist[['Orig.Country', 'Dest.Country', 'Orig', 'Dest', 'Distance (km)']].drop_duplicates()

    # Create a DataFrame for all future predictions (2020-2031)
    years = range(2020, 2032)
    months = range(1, 13)
    future_rows = []
    for year in years:
        for month in months:
            for _, route in unique_routes.iterrows():
                future_rows.append({
                    'year': year, 'month': month, 'Orig.Country': route['Orig.Country'],
                    'Dest.Country': route['Dest.Country'], 'Orig': route['Orig'],
                    'Dest': route['Dest'], 'Distance (km)': route['Distance (km)']
                })
    data_to_predict = pd.DataFrame(future_rows)

    # --- Interpolate External Data (Population, GDP, Tourism) ---
    print("Interpolating population, GDP, and tourism data...")
    pop_data = pd.read_excel('../data/update_pop (1).xlsx')
    gdp_data = pd.read_excel('../data/gdp_update.xlsx')
    tourism_data = pd.read_excel('../data/tourism.xlsx')

    # Interpolate (assuming 2018/2019 data is available)
    pop_preds = interpolate_future_data(pop_data, 2020, 2031, '2019', '2018', 1)
    gdp_preds = interpolate_future_data(gdp_data, 2020, 2031, '2019', '2018', 1)
    tourism_preds = interpolate_future_data(tourism_data, 2020, 2031, '2019', '2018', 1)
    
    # Create dictionaries for fast lookup
    pop_dicts = {str(y): pop_preds.set_index('Country Name')[str(y)].to_dict() for y in years}
    gdp_dicts = {str(y): gdp_preds.set_index('Country')[str(y)].to_dict() for y in years}
    tourism_dicts = {str(y): tourism_preds.set_index('Country Name')[str(y)].to_dict() for y in years}

    # Merge interpolated data into the prediction frame
    data_to_predict['origin_population'] = data_to_predict.apply(lambda r: pop_dicts[str(r['year'])].get(r['Orig.Country']), axis=1)
    data_to_predict['dest_population'] = data_to_predict.apply(lambda r: pop_dicts[str(r['year'])].get(r['Dest.Country']), axis=1)
    data_to_predict['Origin_gdp'] = data_to_predict.apply(lambda r: gdp_dicts[str(r['year'])].get(r['Orig.Country']), axis=1)
    data_to_predict['Dest_gdp'] = data_to_predict.apply(lambda r: gdp_dicts[str(r['year'])].get(r['Dest.Country']), axis=1)
    data_to_predict['Origin_tourism_arrival'] = data_to_predict.apply(lambda r: tourism_dicts[str(r['year'])].get(r['Orig.Country']), axis=1)
    data_to_predict['Dest_tourism_arrival'] = data_to_predict.apply(lambda r: tourism_dicts[str(r['year'])].get(r['Dest.Country']), axis=1)

    data_to_predict = data_to_predict.fillna(0) # Fill any missing lookups with 0

    # --- Iterative Prediction Loop ---
    print("Starting iterative prediction for 2020-2031...")
    
    # Use last known month of historical data (e.g., Dec 2019) as the starting point
    final_historical_data = data_hist[data_hist['Date'] == data_hist['Date'].max()].copy()
    all_predictions = [final_historical_data]

    for year in tqdm(years, desc="Predicting Years"):
        for month in range(1, 13):
            # Get the data for the current month to predict
            current_month_df = data_to_predict[(data_to_predict['year'] == year) & (data_to_predict['month'] == month)].copy()
            if current_month_df.empty:
                continue

            # --- Calculate Prev_passenger for the current month ---
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            
            # Get last month's data (either historical or previously predicted)
            last_month_data = all_predictions[-1][(all_predictions[-1]['year'] == prev_year) & (all_predictions[-1]['month'] == prev_month)]
            
            # Create a lookup dictionary for previous passengers
            prev_pax_lookup = last_month_data.set_index(['Orig', 'Dest'])['Reported.Est..Pax'].to_dict()
            
            current_month_df['Prev_passenger'] = current_month_df.apply(
                lambda row: prev_pax_lookup.get((row['Orig'], row['Dest']), 0), axis=1
            )
            
            # --- Prepare Data for Models ---
            prediction_df_scaled = current_month_df.copy()
            
            # Cyclical month features
            prediction_df_scaled['sin_month'] = np.sin(2 * np.pi * prediction_df_scaled['month'] / 12)
            prediction_df_scaled['cos_month'] = np.cos(2 * np.pi * prediction_df_scaled['month'] / 12)

            # Apply scaling and encoding
            for feature, scaler in scalers.items():
                if feature in prediction_df_scaled.columns and feature != 'Reported.Est..Pax':
                    prediction_df_scaled[feature] = scaler.transform(prediction_df_scaled[[feature]])

            for feature, encoder in label_encoders.items():
                if feature in prediction_df_scaled.columns:
                    # Handle unseen labels by mapping them to a known label (e.g., the first one)
                    known_labels = set(encoder.classes_)
                    prediction_df_scaled[feature] = prediction_df_scaled[feature].apply(lambda x: x if x in known_labels else encoder.classes_[0])
                    prediction_df_scaled[feature] = encoder.transform(prediction_df_scaled[feature])
            
            # --- Make Predictions ---
            # XGBoost Prediction
            xgb_features = xgb_model.get_booster().feature_names
            y_pred_xgb_scaled = xgb_model.predict(prediction_df_scaled[xgb_features])
            
            # Transformer Prediction
            cat_features = ['Orig.Country', 'Dest.Country', 'Orig', 'Dest', 'month']
            num_features = [
                'Distance (km)', 'origin_population', 'dest_population',
                'Origin_gdp', 'Dest_gdp', 'Origin_tourism_arrival',
                'Dest_tourism_arrival', 'Prev_passenger'
            ]
            X_cat_pred = [prediction_df_scaled[col] for col in cat_features]
            X_num_pred = prediction_df_scaled[num_features].values
            y_pred_trans_scaled = transformer_model.predict(X_cat_pred + [X_num_pred])
            
            # --- Combine and Descale ---
            # Weighted average (70% XGBoost, 30% Transformer as in notebook)
            final_pred_scaled = 0.7 * y_pred_xgb_scaled + 0.3 * y_pred_trans_scaled.flatten()
            
            # Descale the final prediction
            final_pred_descaled = scalers['Reported.Est..Pax'].inverse_transform(final_pred_scaled.reshape(-1, 1))

            current_month_df['Reported.Est..Pax'] = np.abs(final_pred_descaled.astype(int))
            all_predictions.append(current_month_df)

    # --- Finalize and Save ---
    final_df = pd.concat(all_predictions, ignore_index=True)
    
    # Filter out the initial historical data to only keep predictions
    final_df = final_df[final_df['year'] >= 2020]
    
    output_path = '../predictions/future_predictions_2020_2031.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Future predictions saved to {output_path}")

if __name__ == '__main__':
    main()