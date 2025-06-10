# src/03_train_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def main():
    print("Loading data for XGBoost training...")
    data = pd.read_csv('../data/data_with_features.csv')
    data = data.dropna() # Drop rows with any missing values from merges

    # --- Feature Engineering & Selection ---
    # Cyclical month features
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

    categorical_features = ['Orig.Country', 'Dest.Country', 'Orig', 'Dest']
    numerical_features = [
        'Distance (km)', 'origin_population', 'dest_population',
        'Origin_gdp', 'Dest_gdp', 'Origin_tourism_arrival',
        'Dest_tourism_arrival', 'Prev_passenger'
    ]
    target = 'Reported.Est..Pax'

    # --- Scaling and Encoding ---
    print("Applying scaling and encoding...")
    scalers = {}
    for feature in numerical_features + [target]:
        scaler = RobustScaler()
        data[feature] = scaler.fit_transform(data[[feature]])
        scalers[feature] = scaler

    encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
        encoders[feature] = encoder

    # Save scalers and encoders
    with open('../models/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    with open('../models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Scalers and encoders saved to ../models/")

    # --- Train/Test Split ---
    features = numerical_features + categorical_features + ['sin_month', 'cos_month']
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- XGBoost Model Training ---
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Save the model
    model.save_model('../models/xgb_model.json')
    print("XGBoost model saved to ../models/xgb_model.json")

    # --- Evaluation ---
    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get actual values
    y_test_descaled = scalers[target].inverse_transform(y_test.values.reshape(-1, 1))
    y_pred_descaled = scalers[target].inverse_transform(y_pred_scaled.reshape(-1, 1))

    mae = mean_absolute_error(y_test_descaled, y_pred_descaled)
    rmse = np.sqrt(np.mean((y_test_descaled - y_pred_descaled)**2))
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    
    # --- Feature Importance Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig('../plots/xgboost_feature_importance.pdf')
    print("Feature importance plot saved to ../plots/")

if __name__ == '__main__':
    main()