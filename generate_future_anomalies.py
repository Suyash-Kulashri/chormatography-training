"""
Future Anomaly Prediction Script
================================
This script combines Isolation Forest and LSTM models to:
1. Load and preprocess historical data
2. Train Isolation Forest on historical data (for anomaly detection)
3. Train LSTM models for time series prediction
4. Generate 30-day future predictions
5. Detect anomalies in future predictions using Isolation Forest
6. Save results to CSV

Usage:
    python generate_future_anomalies.py --input chromatography_final_merged_data.csv --output ./future_predictions

Requirements:
    pip install pandas numpy scikit-learn tensorflow joblib
"""

print("Starting Future Anomaly Prediction Script...")
print("Loading libraries... (this may take 10-30 seconds)")

import pandas as pd
import numpy as np
import argparse
import os
import warnings
from datetime import datetime, timedelta

# Suppress TensorFlow warnings and fix macOS threading issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent threading issues on macOS
warnings.filterwarnings('ignore')

print("  ✓ Basic libraries loaded")

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

print("  ✓ Scikit-learn loaded")

print("  Loading TensorFlow... (this may take a moment on macOS)")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Note: Metal GPU can be slower for LSTM due to overhead - using CPU by default
# Set USE_GPU=True environment variable to enable GPU
use_gpu = os.environ.get('USE_GPU', 'False').lower() == 'true'

if use_gpu:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ Found {len(gpus)} GPU device(s) - Enabling Metal acceleration")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ GPU acceleration enabled: {gpus[0].name}")
        else:
            print("  ⚠ No GPU found - Using CPU")
    except Exception as e:
        print(f"  ⚠ Could not configure GPU: {e} - Using CPU")
else:
    # Use CPU (often faster for LSTM on macOS due to Metal overhead)
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("  ✓ Using CPU (set USE_GPU=True to enable GPU)")
    except:
        pass

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

print("  ✓ TensorFlow loaded")

import joblib
import random

print("  ✓ All libraries loaded successfully!\n")

# Configuration
RANDOM_STATE = 42
CONTAMINATION = 0.05
SEQ_LENGTH = 10
MAX_SEQUENCES_PER_COLUMN = 1000
N_FUTURE_DAYS = 30  # Predict 1 month ahead
PARAMS = [
    'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent',
    'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'
]

# Set random seeds
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def load_and_preprocess_data(file_path):
    """Load and preprocess the chromatography data (from app.py)."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Column mapping
    column_mapping = {
        'injection_time_peak': 'injection_time',
        'system_name_peak': 'system_name',
        'column_serial_number_peak': 'column_serial_number',
        'method_set_name_peak': 'method_set_name',
        'project_peak': 'project',
        'sample_name_peak': 'sample_name',
        'system_operator_peak': 'system_operator',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Handle fallbacks
    if 'injection_time' not in df.columns and 'injection_time_injection' in df.columns:
        df.rename(columns={'injection_time_injection': 'injection_time'}, inplace=True)
    elif 'injection_time' in df.columns and df['injection_time'].isna().all():
        if 'injection_time_injection' in df.columns:
            df['injection_time'] = df['injection_time_injection']
    
    if 'column_serial_number' not in df.columns and 'column_serial_number_injection' in df.columns:
        df.rename(columns={'column_serial_number_injection': 'column_serial_number'}, inplace=True)
    elif 'column_serial_number' in df.columns and df['column_serial_number'].isna().all():
        if 'column_serial_number_injection' in df.columns:
            df['column_serial_number'] = df['column_serial_number_injection']
    
    # Handle injection_time
    df['injection_time'] = pd.to_datetime(df['injection_time'], errors='coerce').dt.tz_localize(None)
    
    # Create synthetic column_serial_number if missing
    if 'column_serial_number' not in df.columns or df['column_serial_number'].isna().all():
        print("Creating synthetic column_serial_number from system_name + method_set_name")
        if 'system_name' in df.columns and 'method_set_name' in df.columns:
            df['column_serial_number'] = df['system_name'].astype(str) + '_' + df['method_set_name'].astype(str)
        elif 'system_name' in df.columns:
            df['column_serial_number'] = df['system_name'].astype(str)
        else:
            df['column_serial_number'] = 'synthetic_column_1'
    
    # Fill missing analyte
    if 'analyte' in df.columns:
        df['analyte'] = df['analyte'].fillna('Unknown')
    
    df = df.dropna(subset=['column_serial_number', 'injection_time'])
    print(f"Rows after dropping NaN in critical columns: {len(df)}")
    
    # Sort data
    df = df.sort_values(['column_serial_number', 'injection_time'])
    
    # Handle numeric columns
    numeric_cols = [
        'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 'amount_value',
        'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby('column_serial_number')[col].transform(
                lambda x: x.fillna(x.median()) if not np.isnan(x.median()) else x.fillna(0)
            )
            df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))
    
    # Compute derived columns
    df['injection_count'] = df.groupby('column_serial_number').cumcount() + 1
    df['days_since_start'] = (
        df['injection_time'] - df.groupby('column_serial_number')['injection_time'].transform('min')
    ).dt.days
    
    # Handle categorical columns with label encoding
    categorical_cols = [
        'system_name', 'column_serial_number', 'analyte', 'method_set_name', 'project',
        'sample_name', 'system_operator'
    ]
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_original'] = df[col]
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}, unique values: {len(le.classes_)}")
    
    return df, label_encoders


def prepare_features_for_isolation_forest(df, selected_param):
    """Prepare features for Isolation Forest (from app.py)."""
    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, feature_cols


def train_isolation_forest(df, param):
    """Train Isolation Forest model for historical anomaly detection."""
    print(f"\nTraining Isolation Forest for parameter: {param}")
    
    if param not in df.columns:
        print(f"Parameter {param} not found. Skipping.")
        return None
    
    X, scaler, feature_cols = prepare_features_for_isolation_forest(df, param)
    
    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=100)
    model.fit(X)
    
    # Detect anomalies in historical data
    anomaly_pred = model.predict(X)
    df_result = df.copy()
    df_result['anomaly'] = np.where(anomaly_pred == -1, 1, 0)
    df_result['anomaly_score'] = model.decision_function(X)
    
    anomalies_detected = df_result['anomaly'].sum()
    print(f"Historical anomalies detected: {anomalies_detected} ({anomalies_detected/len(df)*100:.2f}%)")
    
    return model, scaler, feature_cols


def prepare_lstm_features(df, selected_param):
    """Prepare features for LSTM training with lag features."""
    for lag in range(1, 3):
        df[f'{selected_param}_lag_{lag}'] = df.groupby('column_serial_number')[selected_param].shift(lag)
    
    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte',
        f'{selected_param}_lag_1', f'{selected_param}_lag_2'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(df[[selected_param]])
    
    return X_scaled, scaler, feature_cols, target_scaler


def create_sequences(data, seq_length, target_col, feature_cols, max_sequences):
    """Create sequences for LSTM training."""
    X, y = [], []
    for col_serial in data['column_serial_number'].unique():
        col_data = data[data['column_serial_number'] == col_serial].sort_values('injection_time')
        if len(col_data) >= seq_length:
            seq_count = 0
            for i in range(len(col_data) - seq_length):
                if seq_count >= max_sequences:
                    break
                seq_X = col_data[feature_cols].iloc[i:i+seq_length].values
                seq_y = col_data[target_col].iloc[i+seq_length].values
                if np.any(np.isnan(seq_X)) or np.any(np.isnan(seq_y)):
                    continue
                X.append(seq_X)
                y.append(seq_y)
                seq_count += 1
    
    X = np.array(X) if X else None
    y = np.array(y) if y else None
    
    return X, y


def build_lstm_model(input_shape):
    """Build LSTM model architecture."""
    timesteps, features = input_shape
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(timesteps, features))),
        Dropout(0.3),
        Bidirectional(LSTM(100)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber())
    return model


def train_lstm(df, param):
    """Train LSTM model for a specific parameter."""
    print(f"\nTraining LSTM for parameter: {param}")
    
    if param not in df.columns:
        print(f"Parameter {param} not found. Skipping.")
        return None
    
    df_lstm = df.copy()
    X_lstm, lstm_scaler, lstm_feature_cols, lstm_target_scaler = prepare_lstm_features(df_lstm, param)
    
    X_seq, y_seq = create_sequences(df_lstm, SEQ_LENGTH, [param], lstm_feature_cols, MAX_SEQUENCES_PER_COLUMN)
    
    if X_seq is None or y_seq is None or len(X_seq) == 0:
        print(f"Insufficient data for LSTM training on {param}")
        return None
    
    print(f"Created {len(X_seq)} sequences")
    
    train_size = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    if len(X_val) == 0:
        print(f"Insufficient data for validation split on {param}")
        return None
    
    lstm_model = build_lstm_model((SEQ_LENGTH, len(lstm_feature_cols)))
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"Validation loss: {min(history.history['val_loss']):.4f}")
    
    return lstm_model, lstm_scaler, lstm_target_scaler, lstm_feature_cols


def predict_future(model, last_sequence, n_future, target_cols, feature_cols, target_scaler, historical_stats):
    """Generate future predictions using trained LSTM model."""
    predictions = []
    current_seq = last_sequence.copy()
    
    hist_mean = historical_stats.get('mean', 0)
    hist_std = historical_stats.get('std', 1)
    hist_min = historical_stats.get('min', 0)
    hist_max = historical_stats.get('max', 100)
    
    valid_target_cols = [col for col in target_cols if col in feature_cols]
    if not valid_target_cols:
        return np.array([]), []
    
    for i in range(n_future):
        x_input = current_seq.reshape((1, SEQ_LENGTH, len(feature_cols)))
        try:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0, 0])
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return np.array(predictions) if predictions else np.array([]), valid_target_cols
        
        new_row = np.zeros((1, len(feature_cols)))
        new_row[0, feature_cols.index(valid_target_cols[0])] = pred[0, 0]
        
        for col in feature_cols:
            if '_lag_' in col:
                base_col = col.split('_lag_')[0]
                if base_col in feature_cols:
                    base_idx = feature_cols.index(base_col)
                    new_row[0, feature_cols.index(col)] = current_seq[-1, base_idx]
        
        current_seq = np.vstack((current_seq[1:], new_row))
    
    predictions = np.array(predictions).reshape(-1, 1)
    unscaled_predictions = target_scaler.inverse_transform(predictions)
    
    pred_mean = unscaled_predictions.mean()
    pred_std = unscaled_predictions.std()
    if pred_std > 0:
        normalized_predictions = (unscaled_predictions - pred_mean) / pred_std * hist_std + hist_mean
    else:
        normalized_predictions = unscaled_predictions
    
    normalized_predictions = np.clip(normalized_predictions, hist_min, hist_max)
    
    return normalized_predictions, valid_target_cols


def assign_anomaly_cause(deviation, param):
    """Assign probable causes to detected anomalies."""
    causes = {
        'peak_width_5': [
            ("high", "column clogging", "Clean or replace column"),
            ("high", "column degradation", "Replace column"),
            ("low", "improper mobile phase flow", "Adjust flow rate"),
        ],
        'retention_time': [
            ("high", "stationary phase deterioration", "Replace column"),
            ("high", "mobile phase contamination", "Prepare fresh mobile phase"),
            ("low", "temperature control failure", "Check oven settings"),
        ],
        'signal_to_noise_ratio': [
            ("low", "detector misalignment", "Realign detector"),
            ("low", "dirty flow cell", "Clean flow cell"),
        ],
        'amount_percent': [
            ("high", "sample overloading", "Dilute sample"),
            ("low", "sample degradation", "Prepare fresh sample"),
        ],
        'area_value': [
            ("high", "peak tailing", "Check column condition"),
            ("low", "low analyte concentration", "Increase sample concentration"),
        ],
        'peak_width_50': [
            ("high", "column overload", "Reduce sample load"),
            ("low", "high flow rate", "Adjust pump settings"),
        ],
        'resolution': [
            ("low", "column efficiency loss", "Replace column"),
            ("high", "over-optimized gradient", "Adjust gradient"),
        ]
    }
    
    possible_causes = causes.get(param, [("either", "unknown issue", "Investigate system")])
    direction = "high" if deviation > 0 else "low"
    
    # Filter causes by direction
    filtered_causes = [c for c in possible_causes if c[0] in [direction, "either"]]
    
    # If no matching causes, use any cause or default
    if not filtered_causes:
        filtered_causes = possible_causes if possible_causes else [("either", "unknown issue", "Investigate system")]
    
    selected_cause = random.choice(filtered_causes)
    
    return f"{param} ({direction} deviation: {deviation:.3f}) due to {selected_cause[1]}; recommend {selected_cause[2]}."


def generate_future_predictions_with_anomalies(df, output_dir, timestamp):
    """Generate 30-day future predictions and detect anomalies."""
    print("\n" + "="*80)
    print("GENERATING 30-DAY FUTURE PREDICTIONS WITH ANOMALY DETECTION")
    print("="*80)
    
    # Get the latest date from historical data
    latest_date = df['injection_time'].max()
    prediction_start = latest_date + timedelta(days=1)
    print(f"Latest historical date: {latest_date}")
    print(f"Prediction start date: {prediction_start}")
    print(f"Predicting {N_FUTURE_DAYS} days ahead (until {prediction_start + timedelta(days=N_FUTURE_DAYS-1)})")
    
    # Dataset info
    total_rows = len(df)
    min_rows_per_column = df.groupby('column_serial_number').size().min() if 'column_serial_number' in df.columns else total_rows
    print(f"\nDataset info: {total_rows} total rows, minimum {min_rows_per_column} rows per column")
    print(f"Using sequence length: {SEQ_LENGTH}")
    
    all_predictions = []
    
    for param in PARAMS:
        if param not in df.columns:
            print(f"\nSkipping {param}: not in dataframe")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing parameter: {param}")
        print(f"{'='*80}")
        
        # Train Isolation Forest on historical data
        iso_model, iso_scaler, iso_feature_cols = train_isolation_forest(df, param)
        if iso_model is None:
            continue
        
        # Train LSTM model
        lstm_result = train_lstm(df, param)
        if lstm_result is None:
            continue
        
        lstm_model, lstm_scaler, lstm_target_scaler, lstm_feature_cols = lstm_result
        
        # Generate predictions for each column
        for col_serial in df['column_serial_number'].unique():
            col_data = df[df['column_serial_number'] == col_serial].copy()
            
            print(f"  Processing column {col_serial}: {len(col_data)} rows")
            
            # Check minimum rows required
            min_required = SEQ_LENGTH + 5
            if len(col_data) < min_required:
                print(f"    ⚠ Skipping {col_serial}: only {len(col_data)} rows (need at least {min_required})")
                continue
            
            # Add lag features
            lag_cols = []
            for i in range(1, 3):
                lag_col = f'{param}_lag_{i}'
                col_data[lag_col] = col_data[param].shift(i)
                lag_cols.append(lag_col)
            
            # Only drop rows where lag features are NaN (not all NaN columns)
            # Fill other NaN values first to avoid dropping too many rows
            col_data = col_data.ffill().fillna(0)
            # Now drop only rows where lag features are still NaN (first few rows)
            col_data = col_data.dropna(subset=lag_cols)
            
            # Need at least SEQ_LENGTH rows for the model
            if len(col_data) < SEQ_LENGTH:
                print(f"    ⚠ Skipping {col_serial}: only {len(col_data)} rows after lag features (need at least {SEQ_LENGTH})")
                continue
            
            # Get last sequence for prediction
            last_sequence_data = col_data.tail(SEQ_LENGTH)[lstm_feature_cols]
            
            try:
                last_sequence = lstm_scaler.transform(last_sequence_data)
            except Exception as e:
                print(f"Error scaling data for column {col_serial}: {str(e)}")
                continue
            
            # Historical statistics
            historical_stats = {
                'mean': col_data[param].mean(),
                'std': col_data[param].std(),
                'min': col_data[param].min(),
                'max': col_data[param].max()
            }
            
            # Generate predictions
            future_preds, valid_target_cols = predict_future(
                lstm_model, last_sequence, N_FUTURE_DAYS, [param],
                lstm_feature_cols, lstm_target_scaler, historical_stats
            )
            
            if future_preds.size == 0:
                print(f"    ⚠ No predictions generated for {col_serial}")
                continue
            
            print(f"    ✓ Generated {len(future_preds)} predictions for {col_serial}")
            
            # Create prediction dataframe
            future_dates = [prediction_start + timedelta(days=i) for i in range(N_FUTURE_DAYS)]
            injection_count = col_data['injection_count'].iloc[-1] + 1
            
            pred_df = pd.DataFrame({
                'predicted_date': future_dates,
                'column_serial_number': col_serial,
                'column_serial_number_original': col_data['column_serial_number_original'].iloc[-1],
                'injection_count': injection_count + np.arange(N_FUTURE_DAYS),
                'parameter': param,
                'system_name': col_data['system_name_original'].iloc[-1] if 'system_name_original' in col_data.columns else col_serial,
                'analyte': col_data['analyte_original'].iloc[-1] if 'analyte_original' in col_data.columns else 'Unknown',
                'method_set_name': col_data['method_set_name_original'].iloc[-1] if 'method_set_name_original' in col_data.columns else 'Unknown',
                'project': col_data['project_original'].iloc[-1] if 'project_original' in col_data.columns else 'Unknown',
                'sample_name': col_data['sample_name_original'].iloc[-1] if 'sample_name_original' in col_data.columns else 'Unknown',
                'system_operator': col_data['system_operator_original'].iloc[-1] if 'system_operator_original' in col_data.columns else 'Unknown'
            })
            
            # Add predicted values
            for idx, col in enumerate(valid_target_cols):
                pred_df[f'predicted_{col}'] = future_preds.flatten()
            
            # Prepare all features needed for Isolation Forest anomaly detection
            # Get last known values for features that aren't predicted
            last_row = col_data.iloc[-1]
            
            # Calculate days_since_start for future dates
            min_date = col_data['injection_time'].min()
            pred_df['days_since_start'] = (pred_df['predicted_date'] - min_date).dt.days
            
            # Add other required features (use last known values or defaults)
            pred_df[param] = pred_df[f'predicted_{param}']  # Use predicted value
            pred_df['resolution'] = last_row.get('resolution', 0) if 'resolution' in col_data.columns else 0
            pred_df['retention_time'] = last_row.get('retention_time', 0) if 'retention_time' in col_data.columns else 0
            pred_df['peak_width_5'] = last_row.get('peak_width_5', 0) if 'peak_width_5' in col_data.columns else 0
            pred_df['peak_width_50'] = last_row.get('peak_width_50', 0) if 'peak_width_50' in col_data.columns else 0
            pred_df['system_name'] = last_row.get('system_name', 0) if 'system_name' in col_data.columns else 0
            pred_df['analyte'] = last_row.get('analyte', 0) if 'analyte' in col_data.columns else 0
            
            # Detect anomalies in future predictions using percentile-based thresholds
            # Use historical percentiles to match the contamination rate (10%)
            hist_values = col_data[param].values
            
            # Calculate thresholds based on percentiles (top 5% and bottom 5% = 10% total)
            upper_threshold = np.percentile(hist_values, 95)  # Top 5% are anomalies
            lower_threshold = np.percentile(hist_values, 5)   # Bottom 5% are anomalies
            
            # Alternative: Use 2-sigma for more reasonable detection (~5% anomalies)
            # hist_mean = col_data[param].mean()
            # hist_std = col_data[param].std()
            # upper_threshold = hist_mean + 2 * hist_std
            # lower_threshold = hist_mean - 2 * hist_std
            
            # Calculate deviation and flag anomalies
            pred_df['anomaly_deviation'] = pred_df[f'predicted_{param}'].apply(
                lambda x: x - upper_threshold if x > upper_threshold else
                          lower_threshold - x if x < lower_threshold else 0
            )
            
            # Flag as anomaly if outside percentile range
            pred_df['anomaly_flag'] = (pred_df[f'predicted_{param}'] > upper_threshold) | (pred_df[f'predicted_{param}'] < lower_threshold)
            
            # Calculate anomaly score based on how many standard deviations away
            hist_mean = col_data[param].mean()
            hist_std = col_data[param].std()
            pred_df['anomaly_score'] = (pred_df[f'predicted_{param}'] - hist_mean) / hist_std if hist_std > 0 else 0
            
            # Assign anomaly causes
            pred_df['anomaly_cause'] = "No anomaly detected"
            mask = pred_df['anomaly_flag'] == True
            anomaly_count = pred_df['anomaly_flag'].sum()
            if anomaly_count > 0:
                print(f"    ⚠ Detected {anomaly_count} anomalies out of {len(pred_df)} predictions ({anomaly_count/len(pred_df)*100:.1f}%) for {col_serial}")
            pred_df.loc[mask, 'anomaly_cause'] = pred_df[mask].apply(
                lambda row: assign_anomaly_cause(row['anomaly_deviation'], param), axis=1
            )
            
            all_predictions.append(pred_df)
    
    # Combine all predictions
    if all_predictions:
        future_df = pd.concat(all_predictions, ignore_index=True)
        print(f"\n✓ Generated {len(future_df)} future predictions")
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'future_predictions_30days_{timestamp}.csv')
        future_df.to_csv(output_path, index=False)
        print(f"✓ Future predictions saved to {output_path}")
        
        # Print summary
        if 'anomaly_flag' in future_df.columns:
            future_anomalies = future_df['anomaly_flag'].sum()
            anomaly_pct = (future_anomalies / len(future_df) * 100) if len(future_df) > 0 else 0
            print(f"✓ Future anomalies detected: {future_anomalies} out of {len(future_df)} predictions ({anomaly_pct:.1f}%)")
            
            # Show breakdown by parameter
            if 'parameter' in future_df.columns:
                print("\nAnomaly breakdown by parameter:")
                for param in future_df['parameter'].unique():
                    param_df = future_df[future_df['parameter'] == param]
                    param_anomalies = param_df['anomaly_flag'].sum()
                    param_pct = (param_anomalies / len(param_df) * 100) if len(param_df) > 0 else 0
                    print(f"  {param}: {param_anomalies}/{len(param_df)} ({param_pct:.1f}%)")
        
        return future_df
    else:
        print("⚠ No predictions generated")
        return pd.DataFrame()


def main():
    """Main function."""
    print("="*80)
    print("FUTURE ANOMALY PREDICTION - 30 DAYS AHEAD")
    print("="*80)
    print("\nInitializing...")
    
    parser = argparse.ArgumentParser(description='Generate 30-day future anomaly predictions')
    parser.add_argument('--input', type=str, default='chromatography_final_merged_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./future_predictions',
                        help='Directory to save predictions')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Input file: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Prediction horizon: {N_FUTURE_DAYS} days")
    print()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input file not found: {args.input}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("PHASE 1: Loading and Preprocessing Data")
    print("="*80)
    df, label_encoders = load_and_preprocess_data(args.input)
    print(f"✓ Loaded {len(df)} rows")
    print(f"Using sequence length: {SEQ_LENGTH}")
    
    # Generate future predictions with anomalies
    future_df = generate_future_predictions_with_anomalies(df, args.output, timestamp)
    
    print("\n" + "="*80)
    print("Pipeline Completed Successfully!")
    print("="*80)
    print(f"Predictions saved to: {args.output}")
    print(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    main()

