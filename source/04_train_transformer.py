# src/04_train_transformer.py

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LayerNormalization, Dropout,
    MultiHeadAttention, Add, Flatten, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Helper function for the Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """A single Transformer encoder block."""
    # Attention and Normalization
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward Part
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def main():
    print("Loading data for Transformer training...")
    # Load the data with engineered features
    df = pd.read_csv('../data/data_with_features.csv')
    df = df.dropna().reset_index(drop=True)

    # Load the pre-fitted scalers and encoders
    with open('../models/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open('../models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    print("Loaded pre-fitted scalers and encoders.")

    # --- Define Features and Target ---
    target = 'Reported.Est..Pax'
    categorical_features = ['Orig.Country', 'Dest.Country', 'Orig', 'Dest', 'month']
    numerical_features = [
        'Distance (km)', 'origin_population', 'dest_population',
        'Origin_gdp', 'Dest_gdp', 'Origin_tourism_arrival',
        'Dest_tourism_arrival', 'Prev_passenger'
    ]

    # --- Apply Transformations ---
    print("Applying scaling and encoding to the dataset...")
    for feature in numerical_features + [target]:
        df[feature] = scalers[feature].transform(df[[feature]])

    for feature in categorical_features:
        if feature != 'month': # Month is already numeric 1-12
             df[feature] = label_encoders[feature].transform(df[feature])

    # --- Prepare Data for Model Input ---
    # The Transformer model requires a list of inputs
    from sklearn.model_selection import train_test_split
    
    X_cat = df[categorical_features]
    X_num = df[numerical_features].values
    y = df[target].values

    # Split data before creating the list of inputs
    X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42
    )

    X_train_inputs = [X_cat_train[col] for col in X_cat_train.columns] + [X_num_train]
    X_test_inputs = [X_cat_test[col] for col in X_cat_test.columns] + [X_num_test]

    # --- Build the Transformer Model ---
    print("Building the Transformer model architecture...")
    embedding_input_dims = {
        feature: len(df[feature].unique()) for feature in categorical_features
    }
    embedding_dim = 32 # Embedding dimension for categorical features
    
    # Input Layers
    input_cat_layers = [Input(shape=(1,), name=f"{feat}_input") for feat in categorical_features]
    input_num_layer = Input(shape=(len(numerical_features),), name="numerical_input")
    
    # Embedding Layers
    embedding_layers = [
        Embedding(
            input_dim=embedding_input_dims[cat_feature],
            output_dim=embedding_dim,
            name=f'{cat_feature}_embedding'
        )(input_cat_layers[i])
        for i, cat_feature in enumerate(categorical_features)
    ]
    
    # Flatten embedding outputs
    flattened_embeddings = [Flatten()(emb) for emb in embedding_layers]

    # Numerical feature layer
    dense_num = Dense(embedding_dim, activation='relu', name='numerical_dense')(input_num_layer)

    # Concatenate all features
    merged = concatenate(flattened_embeddings + [dense_num])

    # Reshape for Transformer: (batch_size, sequence_length, feature_dimension)
    # We treat all concatenated features as a single item in a sequence.
    num_features = len(categorical_features) + 1 # +1 for the dense numerical block
    reshaped_inputs = tf.keras.layers.Reshape((num_features, embedding_dim))(merged)

    # Transformer Encoder Layers
    x = reshaped_inputs
    for _ in range(4): # Number of transformer blocks
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)

    # Final Classification Head
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='linear')(x) # Regression output

    model = Model(inputs=input_cat_layers + [input_num_layer], outputs=output)
    model.summary()

    # --- Compile and Train ---
    print("Compiling and training the model...")
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train_inputs, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1
    )
    
    # --- Save the Model ---
    model.save('../models/transformer_model.keras')
    print("Transformer model saved to ../models/transformer_model.keras")

    # --- Evaluation and Plotting ---
    print("Evaluating model and plotting history...")
    # Get the target scaler
    target_scaler = scalers[target]

    # Descale for plotting
    train_mae_descaled = target_scaler.inverse_transform(np.array(history.history['mae']).reshape(-1, 1))
    val_mae_descaled = target_scaler.inverse_transform(np.array(history.history['val_mae']).reshape(-1, 1))

    plt.figure(figsize=(12, 5))

    # Plotting MAE
    plt.subplot(1, 2, 1)
    plt.plot(val_mae_descaled, label='Validation MAE')
    plt.plot(train_mae_descaled, label='Training MAE')
    plt.title('Training and Validation MAE (Descaled)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)

    # Plotting Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../plots/transformer_training_history.pdf')
    print("Training history plot saved to ../plots/")

if __name__ == '__main__':
    main()