"""
Configuration file for Chromatography Anomaly Detection System
================================================================
Centralized configuration to avoid hard-coded values and ensure consistency
across all scripts (train.py, generate_future_anomalies.py, app.py)
"""

import os
from pathlib import Path
import pandas as pd

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
FUTURE_PREDICTIONS_DIR = BASE_DIR / "future_predictions"

# Data files
DEFAULT_CSV_FILE = "chromatography_final_merged_data.csv"
SAMPLE_CSV_FILE = "chromatography_sample_200.csv"

# Model directories
ISOLATION_FOREST_DIR = MODELS_DIR / "isolation_forest"
LSTM_DIR = MODELS_DIR / "lstm"
ENCODERS_DIR = MODELS_DIR / "encoders"
RESULTS_DIR = MODELS_DIR / "results"
MLRUNS_DIR = MODELS_DIR / "mlruns"

# Create directories if they don't exist
for directory in [MODELS_DIR, ISOLATION_FOREST_DIR, LSTM_DIR, ENCODERS_DIR, 
                  RESULTS_DIR, FUTURE_PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model hyperparameters (standardized across all scripts)
RANDOM_STATE = 42
CONTAMINATION = 0.1  # 10% anomaly rate (standardized)
INJECTION_THRESHOLD = 1000
SEQ_LENGTH = 10  # LSTM sequence length
MAX_SEQUENCES_PER_COLUMN = 1000
N_ESTIMATORS = 100  # Isolation Forest trees

# Prediction settings
N_FUTURE_DAYS = 30  # Days to predict ahead
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 10  # Early stopping patience
LSTM_LEARNING_RATE = 0.0005

# Parameters to monitor
PARAMS = [
    'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent',
    'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
    'peak_width_10'
]

# Categorical columns for encoding
CATEGORICAL_COLS = [
    'system_name', 'column_serial_number', 'analyte', 'method_set_name', 
    'project', 'sample_name', 'system_operator'
]

# Numeric columns
NUMERIC_COLS = [
    'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 
    'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
    'peak_width_10'
]

# Feature columns for models
def get_feature_cols(selected_param):
    """Get feature columns for a specific parameter."""
    return [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte'
    ]

# Anomaly threshold method (standardized)
ANOMALY_THRESHOLD_METHOD = "mean_std"  # Options: "mean_std", "percentile"
ANOMALY_STD_MULTIPLIER = 3  # For mean ± N*std method
ANOMALY_PERCENTILE_LOWER = 5  # For percentile method
ANOMALY_PERCENTILE_UPPER = 95  # For percentile method

# Column replacement threshold (injection count)
REPLACEMENT_INJECTION_THRESHOLD = 1000  # Suggest replacement after N injections

# MLflow settings
USE_MLFLOW = True

# Streamlit settings
STREAMLIT_LAYOUT = "wide"

def get_latest_model_timestamp(model_type="encoders"):
    """Get the latest timestamp for saved models."""
    if model_type == "encoders":
        dir_path = ENCODERS_DIR
        pattern = "label_encoders_*.pkl"
    elif model_type == "isolation_forest":
        dir_path = ISOLATION_FOREST_DIR
        pattern = "*_model_*.pkl"
    elif model_type == "lstm":
        dir_path = LSTM_DIR
        pattern = "*_lstm_model_*.h5"
    else:
        return None
    
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    # Extract timestamps and return the latest
    timestamps = []
    for f in files:
        parts = f.stem.split('_')
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # Date part
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # Time part
                    timestamp = f"{part}_{parts[i + 1]}"
                    timestamps.append(timestamp)
                    break
    
    if timestamps:
        return max(timestamps)  # Return latest timestamp
    return None

def get_latest_prediction_file():
    """Get the latest future prediction CSV file."""
    files = list(FUTURE_PREDICTIONS_DIR.glob("future_predictions_30days_*.csv"))
    if not files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def get_model_paths(timestamp, param=None):
    """Get model file paths for a given timestamp and parameter."""
    paths = {
        'encoders': ENCODERS_DIR / f"label_encoders_{timestamp}.pkl",
    }
    
    if param:
        paths.update({
            'isolation_forest': ISOLATION_FOREST_DIR / f"{param}_model_{timestamp}.pkl",
            'lstm_model': LSTM_DIR / f"{param}_lstm_model_{timestamp}.h5",
            'lstm_scaler': LSTM_DIR / f"{param}_lstm_scaler_{timestamp}.pkl",
            'lstm_target_scaler': LSTM_DIR / f"{param}_target_scaler_{timestamp}.pkl",
        })
    
    return paths

def validate_dataframe_columns(df, required_columns, script_name=""):
    """
    Validate that required columns exist in the dataframe.
    
    Args:
        df: pandas DataFrame
        required_columns: list of required column names
        script_name: name of calling script for error messages
        
    Returns:
        tuple: (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"WARNING [{script_name}]: Missing required columns: {missing_columns}")
        return False, missing_columns
    
    return True, []

def validate_numeric_columns(df, numeric_columns):
    """
    Validate that numeric columns contain valid numeric data.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of column names that should be numeric
        
    Returns:
        dict: column_name -> percentage of non-numeric values
    """
    issues = {}
    for col in numeric_columns:
        if col in df.columns:
            try:
                # Try to convert to numeric
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                null_pct = numeric_data.isna().sum() / len(df) * 100
                if null_pct > 0:
                    issues[col] = null_pct
            except Exception as e:
                issues[col] = 100.0  # All values are bad
    
    return issues

def normalize_timezone(dt_series):
    """
    Normalize datetime series to timezone-naive UTC.
    
    Args:
        dt_series: pandas Series with datetime data
        
    Returns:
        pandas Series with timezone-naive datetime
    """
    if pd.api.types.is_datetime64_any_dtype(dt_series):
        if hasattr(dt_series.dt, 'tz') and dt_series.dt.tz is not None:
            return dt_series.dt.tz_convert('UTC').dt.tz_localize(None)
    return dt_series

