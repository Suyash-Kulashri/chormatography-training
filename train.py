"""
Local Training Script for Chromatography Anomaly Detection
============================================================
This script can be run on a local machine without Databricks.
It performs:
1. Data preprocessing
2. Isolation Forest anomaly detection training (historical anomalies)
3. LSTM model training for time series prediction
4. Future prediction generation with anomaly detection
5. Saves models, predictions, and results locally

Usage:
    python train.py --input chromatography_final_merged_data.csv --output ./models

Requirements:
    pip install pandas numpy scikit-learn tensorflow joblib mlflow matplotlib seaborn
"""

print("Starting chromatography training script...")
print("Loading libraries... (this may take 10-30 seconds)")

import pandas as pd
import numpy as np
import argparse
import os
import warnings
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
warnings.filterwarnings('ignore')

print("  ✓ Basic libraries loaded")

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

print("  ✓ Scikit-learn loaded")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

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
import mlflow
import mlflow.sklearn
import mlflow.keras
import random
import json

print("  ✓ All libraries loaded successfully!\n")

# Configuration
RANDOM_STATE = 42
CONTAMINATION = 0.1
INJECTION_THRESHOLD = 1000
SEQ_LENGTH = 10
MAX_SEQUENCES_PER_COLUMN = 1000
N_FUTURE = 14
PARAMS = [
    'analyte','peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent',
    'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'
]

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


class ChromatographyTrainer:
    """Main trainer class for chromatography anomaly detection."""
    
    def __init__(self, input_path, output_dir, use_mlflow=True):
        """
        Initialize the trainer.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save models and results
            use_mlflow: Whether to use MLflow for tracking
        """
        print("Initializing trainer...")
        self.input_path = input_path
        self.output_dir = output_dir
        self.use_mlflow = use_mlflow
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        print(f"Creating output directories in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'isolation_forest'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'lstm'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'encoders'), exist_ok=True)
        print("  ✓ Output directories created")
        
        # Initialize MLflow
        if self.use_mlflow:
            print("Setting up MLflow tracking...")
            # Use absolute path to avoid root directory issues
            mlruns_path = os.path.abspath(os.path.join(output_dir, 'mlruns'))
            mlflow.set_tracking_uri(f"file://{mlruns_path}")
            mlflow.set_experiment("chromatography_local_training")
            print(f"  ✓ MLflow configured: {mlruns_path}")
        
        print("  ✓ Trainer initialized successfully\n")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the chromatography data."""
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Column mapping from new names to expected names
        column_mapping = {
            'injection_time_peak': 'injection_time',
            'system_name_peak': 'system_name',
            'column_serial_number_peak': 'column_serial_number',
            'method_set_name_peak': 'method_set_name',
            'project_peak': 'project',
            'sample_name_peak': 'sample_name',
            'system_operator_peak': 'system_operator',
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Check if fallbacks needed (if _peak columns are empty, use _injection)
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
        
        # Drop unused columns (including _injection fallback columns now that we've used them)
        columns_to_drop = [
            'file_path', 'created_at', 'channel_name', 'usp_tailing', 'asym_at_10',
            'injection_date', 'sample_set_name', 'column_name', 'ids_file_id',
            'manual_integration', 'channel_id', 'peak_start', 'peak_end',
            'symmetry_factor', 'asymmetry_aia', 'asymmetry_usp', 'source_type',
            'sample_type', 'injection_duration', 'custom_field_peak', 'custom_field_injection',
            'source_type_injection', 'injection_id', 'injection_volume', 'result_count',
            'sample_injection_count', 'injection_date_injection', 'year_month_injection',
            'year_month_peak', 'injection_duration_injection', 'project_injection',
            'injection_date_peak', 'sample_set_name_peak',
            'column_name_peak', 'source_type_peak', 'sample_type_peak', 'injection_duration_peak',
            # Drop the _injection fallbacks now that we've used them
            'injection_time_injection', 'column_serial_number_injection',
            'system_name_injection', 'system_operator_injection', 'column_name_injection',
            'sample_name_injection', 'sample_type_injection', 'sample_set_name_injection',
            'method_set_name_injection'
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Handle injection_time
        df["injection_time"] = pd.to_datetime(df["injection_time"], errors="coerce")
        
        # Create synthetic column_serial_number if missing or empty
        if 'column_serial_number' not in df.columns or df['column_serial_number'].isna().all():
            print("Creating synthetic column_serial_number from system_name + method_set_name")
            if 'system_name' in df.columns and 'method_set_name' in df.columns:
                df['column_serial_number'] = df['system_name'].astype(str) + '_' + df['method_set_name'].astype(str)
            elif 'system_name' in df.columns:
                df['column_serial_number'] = df['system_name'].astype(str)
            else:
                df['column_serial_number'] = 'synthetic_column_1'
        
        # Drop rows with missing critical columns
        df = df.dropna(subset=["column_serial_number", "injection_time"])
        print(f"Rows after dropping NaN in critical columns: {len(df)}")
        
        # Sort data
        df = df.sort_values(['column_serial_number', 'injection_time'])
        
        # Handle numeric columns
        numeric_cols = [
            "peak_width_5", "retention_time", "signal_to_noise_ratio", "amount_percent", "amount_value",
            "area_percent", "area_value", "peak_width_50", "resolution", "peak_width_10"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Fill NaN with group median
                df[col] = df.groupby("column_serial_number")[col].transform(
                    lambda x: x.fillna(x.median()) if not np.isnan(x.median()) else x.fillna(0)
                )
                # Clip outliers
                df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))
        
        # Compute derived columns
        df["injection_count"] = df.groupby("column_serial_number").cumcount() + 1
        df["days_since_start"] = (
            df["injection_time"] - df.groupby("column_serial_number")["injection_time"].transform("min")
        ).dt.days
        
        # Handle categorical columns with label encoding
        categorical_cols = [
            "system_name", "column_serial_number", "analyte", "method_set_name", "project",
            "sample_name", "system_operator"
        ]
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                df[f"{col}_original"] = df[col]
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                print(f"Encoded {col}, unique values: {len(le.classes_)}")
        
        # Save label encoders
        encoder_path = os.path.join(self.output_dir, 'encoders', f'label_encoders_{self.timestamp}.pkl')
        joblib.dump(label_encoders, encoder_path)
        print(f"Label encoders saved to {encoder_path}")
        
        self.df = df
        self.label_encoders = label_encoders
        
        return df
    
    def prepare_features(self, df, selected_param):
        """Prepare features for model training."""
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
    
    def train_isolation_forest(self, df, param):
        """Train Isolation Forest model for a specific parameter."""
        print(f"\nTraining Isolation Forest for parameter: {param}")
        
        if param not in df.columns:
            print(f"Parameter {param} not found. Skipping.")
            return None
        
        # Prepare features
        X, scaler, feature_cols = self.prepare_features(df, param)
        
        # Train model
        model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=100)
        model.fit(X)
        
        # Detect anomalies
        anomaly_pred = model.predict(X)
        df_result = df.copy()
        df_result['anomaly'] = np.where(anomaly_pred == -1, 1, 0)
        df_result['anomaly_score'] = model.decision_function(X)
        
        # Calculate anomaly statistics
        mean_val = df[param].mean()
        std_val = df[param].std()
        upper_threshold = mean_val + 3 * std_val
        lower_threshold = mean_val - 3 * std_val
        
        df_result['anomaly_feature'] = param
        df_result['anomaly_deviation'] = df[param].apply(
            lambda x: x - upper_threshold if x > upper_threshold else 
                      lower_threshold - x if x < lower_threshold else 0
        )
        
        # Assign anomaly causes
        df_result = self.assign_anomaly_cause(df_result, param)
        
        # Save model
        model_path = os.path.join(self.output_dir, 'isolation_forest', f'{param}_model_{self.timestamp}.pkl')
        joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, model_path)
        print(f"Model saved to {model_path}")
        
        anomalies_detected = df_result['anomaly'].sum()
        print(f"Anomalies detected: {anomalies_detected} ({anomalies_detected/len(df)*100:.2f}%)")
        
        return df_result, model, scaler, feature_cols, anomalies_detected
    
    def assign_anomaly_cause(self, df, selected_param):
        """Assign probable causes to detected anomalies."""
        df['anomaly_cause'] = "No anomaly detected"
        mask = df['anomaly'] == 1
        deviation = df['anomaly_deviation']
        
        causes = {
            'peak_width_5': [
                ("high", "column clogging", "Clean or replace column"),
                ("high", "column degradation", "Replace column"),
                ("low", "improper mobile phase flow", "Adjust flow rate"),
                ("low", "column packing issue", "Inspect column")
            ],
            'peak_width_10': [
                ("high", "column clogging", "Clean or replace column"),
                ("high", "column degradation", "Replace column"),
                ("low", "improper mobile phase flow", "Adjust flow rate"),
                ("low", "column packing issue", "Inspect column")
            ],
            'retention_time': [
                ("high", "stationary phase deterioration", "Replace column"),
                ("high", "mobile phase contamination", "Prepare fresh mobile phase"),
                ("low", "temperature control failure", "Check oven settings"),
                ("low", "pump pressure irregularity", "Service pump")
            ],
            'signal_to_noise_ratio': [
                ("low", "detector misalignment", "Realign detector"),
                ("low", "dirty flow cell", "Clean flow cell"),
                ("high", "electronic noise", "Check grounding"),
                ("high", "lamp intensity issue", "Replace lamp")
            ],
            'amount_percent': [
                ("high", "sample overloading", "Dilute sample"),
                ("high", "injection volume error", "Calibrate injector"),
                ("low", "sample degradation", "Prepare fresh sample"),
                ("low", "detector sensitivity issue", "Adjust detector settings")
            ],
            'amount_value': [
                ("high", "sample overloading", "Dilute sample"),
                ("high", "injection volume error", "Calibrate injector"),
                ("low", "sample degradation", "Prepare fresh sample"),
                ("low", "detector sensitivity issue", "Adjust detector settings")
            ],
            'area_value': [
                ("high", "peak tailing", "Check column condition"),
                ("high", "sample contamination", "Verify sample purity"),
                ("low", "low analyte concentration", "Increase sample concentration"),
                ("low", "detector drift", "Recalibrate detector")
            ],
            'area_percent': [
                ("high", "peak tailing", "Check column condition"),
                ("high", "sample contamination", "Verify sample purity"),
                ("low", "low analyte concentration", "Increase sample concentration"),
                ("low", "detector drift", "Recalibrate detector")
            ],
            'peak_width_50': [
                ("high", "column overload", "Reduce sample load"),
                ("high", "solvent mismatch", "Check mobile phase"),
                ("low", "high flow rate", "Adjust pump settings"),
                ("low", "column damage", "Replace column")
            ],
            'resolution': [
                ("low", "column efficiency loss", "Replace column"),
                ("low", "mobile phase composition error", "Verify eluent"),
                ("high", "over-optimized gradient", "Adjust gradient"),
                ("high", "analyte co-elution", "Modify method")
            ]
        }
        
        possible_causes = causes.get(selected_param, [("either", "unknown issue", "Investigate system")])
        df['direction'] = np.where(deviation > 0, "high", "low")
        
        def get_cause(row):
            if not mask[row.name] or row['anomaly_deviation'] == 0:
                return "No anomaly detected"
            direction = row['direction']
            selected_cause = random.choice([c for c in possible_causes if c[0] in [direction, "either"]])
            cause_text = f"{selected_param} ({direction} deviation: {row['anomaly_deviation']:.3f}) due to {selected_cause[1]}; recommend {selected_cause[2]}."
            return cause_text
        
        df.loc[mask, 'anomaly_cause'] = df[mask].apply(get_cause, axis=1)
        df = df.drop(columns=['direction'], errors='ignore')
        
        return df
    
    def prepare_lstm_features(self, df, selected_param):
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
    
    def create_sequences(self, data, seq_length, target_col, feature_cols, max_sequences):
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
    
    def build_lstm_model(self, input_shape):
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
    
    def train_lstm(self, df, param):
        """Train LSTM model for a specific parameter."""
        print(f"\nTraining LSTM for parameter: {param}")
        
        if param not in df.columns:
            print(f"Parameter {param} not found. Skipping.")
            return None
        
        # Create a copy with lag features for LSTM
        df_lstm = df.copy()
        X_lstm, lstm_scaler, lstm_feature_cols, lstm_target_scaler = self.prepare_lstm_features(df_lstm, param)
        
        # Create sequences (use df_lstm which now has lag columns)
        X_seq, y_seq = self.create_sequences(df_lstm, SEQ_LENGTH, [param], lstm_feature_cols, MAX_SEQUENCES_PER_COLUMN)
        
        if X_seq is None or y_seq is None or len(X_seq) == 0:
            print(f"Insufficient data for LSTM training on {param}")
            return None
        
        print(f"Created {len(X_seq)} sequences")
        
        # Train/validation split
        train_size = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        if len(X_val) == 0:
            print(f"Insufficient data for validation split on {param}")
            return None
        
        # Build and train model
        lstm_model = self.build_lstm_model((SEQ_LENGTH, len(lstm_feature_cols)))
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model and scalers
        model_path = os.path.join(self.output_dir, 'lstm', f'{param}_lstm_model_{self.timestamp}.h5')
        scaler_path = os.path.join(self.output_dir, 'lstm', f'{param}_lstm_scaler_{self.timestamp}.pkl')
        target_scaler_path = os.path.join(self.output_dir, 'lstm', f'{param}_target_scaler_{self.timestamp}.pkl')
        
        lstm_model.save(model_path)
        joblib.dump(lstm_scaler, scaler_path)
        joblib.dump(lstm_target_scaler, target_scaler_path)
        
        print(f"LSTM model saved to {model_path}")
        print(f"Validation loss: {min(history.history['val_loss']):.4f}")
        
        return lstm_model, lstm_scaler, lstm_target_scaler, lstm_feature_cols, history
    
    def predict_future(self, model, last_sequence, n_future, target_cols, feature_cols, target_scaler, historical_stats):
        """Generate future predictions using trained LSTM model."""
        predictions = []
        current_seq = last_sequence.copy()
        
        # Get historical statistics for normalization
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
            
            # Create new row with prediction
            new_row = np.zeros((1, len(feature_cols)))
            new_row[0, feature_cols.index(valid_target_cols[0])] = pred[0, 0]
            
            # Fill in lag features if they exist
            for col in feature_cols:
                if '_lag_' in col:
                    base_col = col.split('_lag_')[0]
                    if base_col in feature_cols:
                        base_idx = feature_cols.index(base_col)
                        new_row[0, feature_cols.index(col)] = current_seq[-1, base_idx]
            
            # Update sequence
            current_seq = np.vstack((current_seq[1:], new_row))
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        unscaled_predictions = target_scaler.inverse_transform(predictions)
        
        # Normalize to match historical statistics
        pred_mean = unscaled_predictions.mean()
        pred_std = unscaled_predictions.std()
        if pred_std > 0:
            normalized_predictions = (unscaled_predictions - pred_mean) / pred_std * hist_std + hist_mean
        else:
            normalized_predictions = unscaled_predictions
        
        # Clip to historical range
        normalized_predictions = np.clip(normalized_predictions, hist_min, hist_max)
        
        return normalized_predictions, valid_target_cols
    
    def assign_anomaly_cause(self, df, selected_param, is_future_prediction=False):
        """Assign cause to detected anomalies."""
        param_key = f'predicted_{selected_param}' if is_future_prediction else selected_param
        if param_key not in df.columns:
            df['anomaly_cause'] = "Unknown"
            return df
        
        df['anomaly_cause'] = "No anomaly detected"
        mask = df['anomaly_flag'] if is_future_prediction else df['anomaly'] == 1
        deviation = df['anomaly_deviation']
        
        # Assign causes based on deviation magnitude
        df.loc[mask & (abs(deviation) > 3), 'anomaly_cause'] = "Extreme deviation"
        df.loc[mask & (abs(deviation) > 2) & (abs(deviation) <= 3), 'anomaly_cause'] = "High deviation"
        df.loc[mask & (abs(deviation) > 1) & (abs(deviation) <= 2), 'anomaly_cause'] = "Moderate deviation"
        df.loc[mask & (abs(deviation) <= 1), 'anomaly_cause'] = "Minor deviation"
        
        return df
    
    def generate_predictions_with_anomalies(self, df, historical_df):
        """Generate future predictions and detect anomalies (Phase 3)."""
        prediction_start = pd.Timestamp('2025-07-01').tz_localize(None)
        n_future = 14  # Predict 2 weeks ahead (14 days)
        
        all_predictions = []
        
        for param in PARAMS:
            if param not in df.columns:
                print(f"Skipping {param}: not in dataframe")
                continue
            
            print(f"\nGenerating future predictions for parameter: {param}")
            
            # Load trained models
            try:
                model_path = os.path.join(self.output_dir, 'lstm', f'{param}_lstm_model_{self.timestamp}.h5')
                scaler_path = os.path.join(self.output_dir, 'lstm', f'{param}_lstm_scaler_{self.timestamp}.pkl')
                target_scaler_path = os.path.join(self.output_dir, 'lstm', f'{param}_target_scaler_{self.timestamp}.pkl')
                
                lstm_model = load_model(model_path)
                lstm_scaler = joblib.load(scaler_path)
                target_scaler = joblib.load(target_scaler_path)
            except Exception as e:
                print(f"Error loading models for {param}: {str(e)}")
                continue
            
            # Get feature columns from scaler
            if hasattr(lstm_scaler, 'feature_names_in_'):
                feature_cols = list(lstm_scaler.feature_names_in_)
            else:
                # Reconstruct feature columns
                feature_cols = [param, 'injection_count', 'days_since_start', 
                               'system_name', 'analyte', 'method_set_name', 'project', 
                               'sample_name', 'system_operator']
                for i in range(1, 4):
                    feature_cols.append(f'{param}_lag_{i}')
            
            # Prepare data with lag features
            df_lstm = df.copy()
            for col_serial in df_lstm['column_serial_number'].unique():
                col_data = df_lstm[df_lstm['column_serial_number'] == col_serial].copy()
                
                if len(col_data) < SEQ_LENGTH + 5:
                    continue
                
                # Add lag features
                for i in range(1, 4):
                    col_data[f'{param}_lag_{i}'] = col_data[param].shift(i)
                
                col_data = col_data.dropna()
                
                if len(col_data) < SEQ_LENGTH:
                    continue
                
                # Get last sequence for prediction
                last_sequence_data = col_data.tail(SEQ_LENGTH)[feature_cols]
                
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
                future_preds, valid_target_cols = self.predict_future(
                    lstm_model, last_sequence, n_future, [param], 
                    feature_cols, target_scaler, historical_stats
                )
                
                if future_preds.size == 0:
                    continue
                
                # Create prediction dataframe
                last_date = col_data['injection_time'].iloc[-1]
                future_dates = [prediction_start + timedelta(days=i) for i in range(n_future)]
                injection_count = col_data['injection_count'].iloc[-1] + 1
                
                pred_df = pd.DataFrame({
                    'column_serial_number': col_serial,
                    'column_serial_number_original': col_data['column_serial_number_original'].iloc[-1],
                    'predicted_date': future_dates,
                    'injection_count': injection_count + np.arange(n_future),
                    'parameter': param,
                    'system_name': col_data['system_name_original'].iloc[-1],
                    'analyte': col_data['analyte_original'].iloc[-1],
                    'method_set_name': col_data['method_set_name_original'].iloc[-1],
                    'project': col_data['project_original'].iloc[-1],
                    'sample_name': col_data['sample_name_original'].iloc[-1],
                    'system_operator': col_data['system_operator_original'].iloc[-1]
                })
                
                # Add predicted values
                for idx, col in enumerate(valid_target_cols):
                    pred_df[f'predicted_{col}'] = future_preds.flatten()
                
                # Detect anomalies using Isolation Forest
                pred_features = [f'predicted_{c}' for c in valid_target_cols]
                if pred_features:
                    X_pred = pred_df[pred_features].values
                    
                    if not np.any(np.isnan(X_pred)):
                        iso_forest = IsolationForest(
                            contamination=CONTAMINATION,
                            random_state=42,
                            n_estimators=100
                        )
                        
                        anomaly_scores = iso_forest.fit_predict(X_pred)
                        pred_df['anomaly_flag'] = anomaly_scores == -1
                        pred_df['anomaly_score'] = iso_forest.decision_function(X_pred)
                        
                        # Calculate deviation
                        pred_mean = future_preds.mean()
                        pred_std = future_preds.std()
                        if pred_std > 0:
                            pred_df['anomaly_deviation'] = (future_preds.flatten() - pred_mean) / pred_std
                        else:
                            pred_df['anomaly_deviation'] = 0
                        
                        # Assign anomaly causes
                        pred_df = self.assign_anomaly_cause(pred_df, valid_target_cols[0], is_future_prediction=True)
                        
                        all_predictions.append(pred_df)
        
        # Combine with historical anomalies
        if all_predictions:
            future_df = pd.concat(all_predictions, ignore_index=True)
            print(f"\n✓ Generated {len(future_df)} future predictions")
            
            if historical_df is not None:
                # Add predicted_date to historical data (use injection_time)
                historical_df['predicted_date'] = historical_df['injection_time']
                # Ensure timezone-naive for comparison
                if pd.api.types.is_datetime64_any_dtype(historical_df['predicted_date']):
                    if historical_df['predicted_date'].dt.tz is not None:
                        historical_df['predicted_date'] = historical_df['predicted_date'].dt.tz_convert('UTC').dt.tz_localize(None)
                
                # Combine historical and future
                final_df = pd.concat([historical_df, future_df], ignore_index=True)
                print(f"✓ Combined {len(historical_df)} historical + {len(future_df)} future = {len(final_df)} total records")
            else:
                final_df = future_df
            
            # Ensure predicted_date is timezone-naive for comparison
            if pd.api.types.is_datetime64_any_dtype(final_df['predicted_date']):
                if final_df['predicted_date'].dt.tz is not None:
                    final_df['predicted_date'] = final_df['predicted_date'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Save final predictions
            final_path = os.path.join(self.output_dir, 'results', f'final_predictions_{self.timestamp}.csv')
            final_df.to_csv(final_path, index=False)
            print(f"✓ Final predictions saved to {final_path}")
            
            # Print summary
            if 'anomaly_flag' in final_df.columns:
                future_anomalies = final_df[final_df['predicted_date'] >= prediction_start]['anomaly_flag'].sum()
                print(f"✓ Future anomalies detected: {future_anomalies}")
            
            return final_df
        else:
            print("⚠ No predictions generated")
            return historical_df
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("="*80)
        print("Starting Chromatography Anomaly Detection Training Pipeline")
        print("="*80)
        
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"full_pipeline_{self.timestamp}"):
                mlflow.log_param("timestamp", self.timestamp)
                mlflow.log_param("input_path", self.input_path)
                mlflow.log_param("contamination", CONTAMINATION)
                mlflow.log_param("seq_length", SEQ_LENGTH)
                
                # Load and preprocess data
                df = self.load_and_preprocess_data()
                mlflow.log_metric("total_rows", len(df))
                
                # Train Isolation Forest models
                print("\n" + "="*80)
                print("PHASE 1: Isolation Forest Anomaly Detection")
                print("="*80)
                
                all_anomalies = []
                for param in PARAMS:
                    result = self.train_isolation_forest(df, param)
                    if result is not None:
                        df_result, model, scaler, feature_cols, anomalies = result
                        all_anomalies.append(df_result)
                        mlflow.log_metric(f"anomalies_{param}", anomalies)
                
                # Save anomaly results
                if all_anomalies:
                    anomaly_df = pd.concat(all_anomalies, ignore_index=True)
                    anomaly_path = os.path.join(self.output_dir, 'results', f'anomalies_{self.timestamp}.csv')
                    anomaly_df.to_csv(anomaly_path, index=False)
                    print(f"\nAnomaly results saved to {anomaly_path}")
                    mlflow.log_artifact(anomaly_path)
                
                # Train LSTM models
                print("\n" + "="*80)
                print("PHASE 2: LSTM Time Series Prediction")
                print("="*80)
                
                for param in PARAMS:
                    result = self.train_lstm(df, param)
                    if result is not None:
                        lstm_model, lstm_scaler, target_scaler, feature_cols, history = result
                        mlflow.log_metric(f"val_loss_{param}", min(history.history['val_loss']))
                
                # Phase 3: Generate Future Predictions
                print("\n" + "="*80)
                print("PHASE 3: Generating Future Predictions")
                print("="*80)
                
                final_df = self.generate_predictions_with_anomalies(df, anomaly_df if all_anomalies else None)
                if final_df is not None:
                    mlflow.log_metric("total_final_rows", len(final_df))
                
                print("\n" + "="*80)
                print("Training Pipeline Completed Successfully!")
                print("="*80)
                print(f"Models saved to: {self.output_dir}")
                print(f"Timestamp: {self.timestamp}")
        else:
            # Same logic without MLflow
            df = self.load_and_preprocess_data()
            
            print("\n" + "="*80)
            print("PHASE 1: Isolation Forest Anomaly Detection")
            print("="*80)
            
            all_anomalies = []
            for param in PARAMS:
                result = self.train_isolation_forest(df, param)
                if result is not None:
                    df_result, model, scaler, feature_cols, anomalies = result
                    all_anomalies.append(df_result)
            
            if all_anomalies:
                anomaly_df = pd.concat(all_anomalies, ignore_index=True)
                anomaly_path = os.path.join(self.output_dir, 'results', f'anomalies_{self.timestamp}.csv')
                anomaly_df.to_csv(anomaly_path, index=False)
                print(f"\nAnomaly results saved to {anomaly_path}")
            
            print("\n" + "="*80)
            print("PHASE 2: LSTM Time Series Prediction")
            print("="*80)
            
            for param in PARAMS:
                self.train_lstm(df, param)
            
            # Phase 3: Generate Future Predictions
            print("\n" + "="*80)
            print("PHASE 3: Generating Future Predictions")
            print("="*80)
            
            self.generate_predictions_with_anomalies(df, anomaly_df if all_anomalies else None)
            
            print("\n" + "="*80)
            print("Training Pipeline Completed Successfully!")
            print("="*80)
            print(f"Models saved to: {self.output_dir}")
            print(f"Timestamp: {self.timestamp}")


def main():
    """Main function to run the training script."""
    print("="*80)
    print("CHROMATOGRAPHY ANOMALY DETECTION - LOCAL TRAINING")
    print("="*80)
    print("\nInitializing... (TensorFlow loading may take a moment)")
    
    parser = argparse.ArgumentParser(description='Train chromatography anomaly detection models locally')
    parser.add_argument('--input', type=str, default='chromatography_final_merged_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./models',
                        help='Directory to save models and results')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Input file: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  MLflow tracking: {'Disabled' if args.no_mlflow else 'Enabled'}")
    print()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input file not found: {args.input}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Available CSV files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"     - {file}")
        return
    
    # Initialize and run trainer
    trainer = ChromatographyTrainer(
        input_path=args.input,
        output_dir=args.output,
        use_mlflow=not args.no_mlflow
    )
    
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()

