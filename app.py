import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Import configuration
import config

# Use configuration from config.py
CSV_FILE = config.DEFAULT_CSV_FILE
MODEL_OUTPUT = str(config.BASE_DIR / "isolation_forest_model.pkl")
RANDOM_STATE = config.RANDOM_STATE
CONTAMINATION = config.CONTAMINATION
INJECTION_THRESHOLD = config.INJECTION_THRESHOLD
CATEGORICAL_COLS = config.CATEGORICAL_COLS
NUMERIC_COLS = config.NUMERIC_COLS

# Get latest prediction file dynamically
PREDICTED_CSV_FILE = config.get_latest_prediction_file()
if PREDICTED_CSV_FILE is None:
    PREDICTED_CSV_FILE = str(config.FUTURE_PREDICTIONS_DIR / "future_predictions_30days_latest.csv")
else:
    PREDICTED_CSV_FILE = str(PREDICTED_CSV_FILE)

# Custom CSS for styling
st.markdown("""
<style>
:root {
    --primary: #2c3e50;
    --secondary: #4f46e5;
    --success: #27ae60;
    --danger: #e74c3c;
    --light: #ecf0f1;
    --dark: #2c3e50;
}
/* Page Title */
.page-title {
    color: var(--primary);
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--secondary);
}
    [data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2c3e50 !important;
    color: white !important;
}
    /* Footer */
.footer {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.9rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{file_path}' not found.")

    # Column mapping from _peak columns to expected names (from train.py)
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
    
    # Handle fallbacks (if _peak columns are empty, use _injection)
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

    expected_columns = [
        'injection_time', 'system_name', 'column_serial_number', 'peak_width_5', 'retention_time', 
        'signal_to_noise_ratio', 'amount_percent', 'amount_value', 'area_percent', 'area_value', 
        'peak_width_50', 'resolution', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ]
    available_columns = df.columns.tolist()
    missing_columns = [col for col in expected_columns if col not in available_columns]
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}. Using available columns: {available_columns}")

    if len(df.columns) != len(df.columns.unique()):
        st.warning("Duplicate column names detected in CSV. Removing duplicates...")
        df = df.loc[:, ~df.columns.duplicated()]

    if 'injection_time' not in df.columns:
        raise ValueError("Required column 'injection_time' is missing.")

    # Handle injection_time with timezone normalization
    df['injection_time'] = pd.to_datetime(df['injection_time'], errors='coerce')
    df['injection_time'] = config.normalize_timezone(df['injection_time'])
    
    # Validate critical columns
    critical_cols = ["injection_time"]
    is_valid, missing = config.validate_dataframe_columns(df, critical_cols, "app.py")
    if not is_valid:
        st.error(f"Missing critical columns: {missing}")
        raise ValueError(f"Missing critical columns: {missing}")
    
    # CRITICAL: Create synthetic column_serial_number FIRST (new_data.csv is 100% missing this)
    if 'column_serial_number' in df.columns and df['column_serial_number'].isna().all():
        df['column_serial_number'] = df['system_name'].astype(str) + '_' + df['method_set_name'].astype(str)
    
    # Fill missing analyte
    if 'analyte' in df.columns:
        df['analyte'] = df['analyte'].fillna('Unknown')
    
    df = df.dropna(subset=['column_serial_number', 'injection_time'])

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby('column_serial_number')[col].transform(
                lambda x: x.fillna(x.median()) if x.median() == x.median() else x.fillna(0)
            )
            df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))

    df = df.sort_values(['column_serial_number', 'injection_time'])
    df['injection_count'] = df.groupby('column_serial_number').cumcount() + 1
    df['days_since_start'] = (df['injection_time'] - df.groupby('column_serial_number')['injection_time'].transform('min')).dt.days

    # Save original values before encoding
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f'{col}_original'] = df[col]

    # Try to load pre-trained label encoders
    latest_timestamp = config.get_latest_model_timestamp("encoders")
    label_encoders = None
    
    if latest_timestamp:
        encoder_path = config.get_model_paths(latest_timestamp)['encoders']
        if encoder_path.exists():
            try:
                label_encoders = joblib.load(encoder_path)
                print(f"Loaded pre-trained encoders from {encoder_path}")
            except Exception as e:
                print(f"Error loading encoders: {e}. Creating new encoders.")
    
    # If no pre-trained encoders, create new ones
    if label_encoders is None:
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    else:
        # Use pre-trained encoders
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                # Handle unseen labels
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return df, label_encoders

def prepare_features(df, selected_param):
    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols

def train_anomaly_model(X):
    # Optimized configuration from config.py (via Grid Search tuning)
    model = IsolationForest(
        contamination=CONTAMINATION, 
        random_state=RANDOM_STATE, 
        n_estimators=config.N_ESTIMATORS,
        max_features=config.MAX_FEATURES,
        max_samples=config.MAX_SAMPLES
    )
    model.fit(X)
    return model

def detect_anomalies(df, X, model, feature_cols, selected_param):
    # Standardized anomaly detection logic
    # Use Isolation Forest for detection, but calculate thresholds using percentile method
    # This ensures consistency with future anomaly detection
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    df['anomaly_score'] = model.decision_function(X)
    
    # Use percentile method for threshold calculation (matches contamination rate ~10%)
    # This ensures consistency between historical and future anomaly detection
    mean_val = df[selected_param].mean()
    std_val = df[selected_param].std()
    upper_threshold = df[selected_param].quantile(config.ANOMALY_PERCENTILE_UPPER / 100)
    lower_threshold = df[selected_param].quantile(config.ANOMALY_PERCENTILE_LOWER / 100)
    
    # Note: Isolation Forest may detect different anomalies than percentile thresholds
    # This is expected - Isolation Forest considers multivariate patterns
    # But we use percentile thresholds for consistency with future predictions
    
    df['anomaly_feature'] = selected_param
    df['anomaly_deviation'] = df[selected_param].apply(
        lambda x: x - upper_threshold if x > upper_threshold else 
                  lower_threshold - x if x < lower_threshold else 0
    )
    
    # Return threshold stats
    return df, {
        'mean': mean_val, 
        'std': std_val, 
        'upper_threshold': upper_threshold, 
        'lower_threshold': lower_threshold
    }

def load_predicted_anomalies(file_path, selected_param, iqr_stats):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Predicted anomalies CSV file '{file_path}' not found.")
        return pd.DataFrame()

    if 'predicted_date' not in df.columns:
        st.error("Required column 'predicted_date' missing in predicted anomalies CSV.")
        return pd.DataFrame()

    df['predicted_date'] = pd.to_datetime(df['predicted_date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['predicted_date'])

    required_cols = [
        f'predicted_{selected_param}', 'anomaly_flag', 'anomaly_cause', 
        'replacement_alert', 'column_serial_number', 'injection_count'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in predicted anomalies CSV: {missing_cols}")

    if f'predicted_{selected_param}' in df.columns:
        upper_threshold = iqr_stats['upper_threshold']
        lower_threshold = iqr_stats['lower_threshold']
        df['anomaly_deviation'] = df[f'predicted_{selected_param}'].apply(
            lambda x: x - upper_threshold if x > upper_threshold else 
                      lower_threshold - x if x < lower_threshold else 0
        )
    else:
        st.warning(f"Column 'predicted_{selected_param}' not found in predicted data. Setting anomaly_deviation to 0.")
        df['anomaly_deviation'] = 0

    df['anomaly_feature'] = selected_param
    
    # Handle NaN values in anomaly_flag column (fix for boolean indexing)
    if 'anomaly_flag' in df.columns:
        df['anomaly_flag'] = df['anomaly_flag'].fillna(False).astype(bool)
    
    return df

def wrap_text(text, width=30):
    return '<br>'.join(textwrap.wrap(text, width=width))

def main():
    st.set_page_config(layout="wide")
    st.markdown('<h1 class="page-title">Chromatography Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("View your chromatography data, detect anomalies, and visualize predictions.")
    
    # Data source information
    st.info(f"📊 **Data Source:** Using YOUR `{CSV_FILE}` directly (old newww.csv used only as parameter reference)")

    try:
        df, label_encoders = load_and_preprocess_data(CSV_FILE)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.sidebar.header("Apply Filters")
    
    # Get actual date range from data
    data_min_date = df['injection_time'].min()
    data_max_date = df['injection_time'].max()
    
    # Check for valid dates
    if pd.isna(data_min_date) or pd.isna(data_max_date):
        st.error("❌ No valid dates found in the data. Please check the 'injection_time' column.")
        return
    
    st.sidebar.info(f"📅 Data available: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}")
    
    # Set intelligent defaults (last 30 days of data)
    default_end = data_max_date
    default_start = default_end - timedelta(days=30)
    if default_start < data_min_date:
        default_start = data_min_date
    
    # Dynamic date pickers
    start_date = st.sidebar.date_input(
        "Training Start Date",
        value=default_start.date(),
        min_value=data_min_date.date(),
        max_value=data_max_date.date()
    )
    end_date = st.sidebar.date_input(
        "Training End Date", 
        value=default_end.date(),
        min_value=data_min_date.date(),
        max_value=data_max_date.date()
    )
    
    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    system_name = st.sidebar.selectbox("System Name", ["All"] + sorted(df['system_name_original'].unique().tolist()))
    method_set = st.sidebar.selectbox("Method Set", ["All"] + sorted(df['method_set_name_original'].unique().tolist()))
    selected_column = st.sidebar.multiselect("Column Serial Number", ["All"] + sorted(df['column_serial_number_original'].unique().tolist()), default=["All"])
    use_pretrained_models = st.sidebar.checkbox("Use Pre-trained Models (faster)", value=True,
                                                  help="Load pre-trained models instead of retraining on filtered data")
    show_predicted_table = st.sidebar.checkbox("Show Predicted Data Table", value=False)
    show_deviation_graph = st.sidebar.checkbox("Show Deviation Graph", value=False)
    show_chromatogram_overlay = st.sidebar.checkbox("Show Chromatogram Overlay", value=False)
    color_by = st.sidebar.selectbox("Color By", [
        'system_name', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ])
    deviation_time_range = st.sidebar.selectbox("Deviation Time Range", ["One Day", "One Week", "One Month"], index=2)

    available_params = [
        col for col in [
            'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 
            'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
            'peak_width_10'
        ] if col in df.columns
    ]
    # All parameters now have predictions available!
    selected_param = st.sidebar.selectbox("Performance Metric", available_params, index=0)

    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    prediction_start = end_date + timedelta(days=1)

    filtered_data = df[(df['injection_time'] >= start_date) & (df['injection_time'] <= end_date)]
    if system_name != "All":
        filtered_data = filtered_data[filtered_data['system_name_original'] == system_name]
    if method_set != "All":
        filtered_data = filtered_data[filtered_data['method_set_name_original'] == method_set]
    if "All" not in selected_column and selected_column:
        filtered_data = filtered_data[filtered_data['column_serial_number_original'].isin(selected_column)]

    if filtered_data.empty:
        st.error("No data found for the selected date range. Please adjust the Training Start Date and Training End Date to include historical data.")
        return

    filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]

    try:
        X, scaler, feature_cols = prepare_features(filtered_data, selected_param)
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return

    # Try to load pre-trained model if checkbox is selected
    iso_model = None
    model_loaded = False
    
    if use_pretrained_models:
        latest_timestamp = config.get_latest_model_timestamp("isolation_forest")
        if latest_timestamp:
            model_path = config.get_model_paths(latest_timestamp, selected_param)['isolation_forest']
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    iso_model = model_data['model']
                    # Use the scaler from saved model for consistency
                    if 'scaler' in model_data:
                        scaler = model_data['scaler']
                    model_loaded = True
                    st.info(f"✅ Loaded pre-trained model for {selected_param} from {model_path.name}")
                except Exception as e:
                    st.warning(f"⚠️ Could not load pre-trained model: {e}. Training new model...")
    
    # Train new model if pre-trained model not loaded
    if iso_model is None:
        iso_model = train_anomaly_model(X)
        if not model_loaded:
            st.info(f"📊 Training new Isolation Forest model for {selected_param}...")
        # Save the newly trained model
        joblib.dump({'model': iso_model, 'scaler': scaler, 'label_encoders': label_encoders}, MODEL_OUTPUT)
    
    filtered_data, iqr_stats = detect_anomalies(filtered_data, X, iso_model, feature_cols, selected_param)
    
    # Show anomaly detection statistics
    total_points = len(filtered_data)
    anomaly_count = (filtered_data['anomaly'] == 1).sum()
    anomaly_pct = (anomaly_count / total_points * 100) if total_points > 0 else 0
    st.info(f"📊 **Anomaly Detection:** {anomaly_count:,} anomalies detected out of {total_points:,} data points ({anomaly_pct:.1f}%) - Expected: ~{CONTAMINATION*100:.0f}%")

    historical_values = filtered_data[selected_param].values
    anomaly_threshold = np.mean(historical_values) + 3 * np.std(historical_values)

    future_anomalies = load_predicted_anomalies(PREDICTED_CSV_FILE, selected_param, iqr_stats)
    
    # Handle NaN values in anomaly_flag column (fix for boolean indexing)
    if not future_anomalies.empty and 'anomaly_flag' in future_anomalies.columns:
        future_anomalies['anomaly_flag'] = future_anomalies['anomaly_flag'].fillna(False).astype(bool)
    
    # DEBUG: Show prediction loading status
    if not future_anomalies.empty:
        st.info(f"✅ Loaded {len(future_anomalies)} predictions for dates {future_anomalies['predicted_date'].min()} to {future_anomalies['predicted_date'].max()}")
    else:
        st.warning("⚠️ No predictions loaded from file!")

    fig = go.Figure()

    color_col = color_by + '_original' if f'{color_by}_original' in filtered_data.columns else color_by
    unique_values = filtered_data[color_col].astype(str).unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    # 1. HISTORICAL DATA - All data points (normal + anomalies)
    # Show all historical data points first (as a baseline)
    fig.add_trace(go.Scatter(
        x=filtered_data['injection_time'],
        y=filtered_data[selected_param],
        mode='markers',
        name='Historical Data',
        marker=dict(color='lightblue', size=6, opacity=0.6),
        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>Type</b>: Historical<extra></extra>',
        legendgroup='historical',
        showlegend=True
    ))

    # 2. HISTORICAL ANOMALIES - Detected anomalies in historical data
    historical_anomalies = filtered_data[filtered_data['anomaly'] == 1]
    if not historical_anomalies.empty:
        fig.add_trace(go.Scatter(
            x=historical_anomalies['injection_time'],
            y=historical_anomalies[selected_param],
            mode='markers',
            name=f'Historical Anomalies ({len(historical_anomalies)})',
            marker=dict(color='orange', symbol='x', size=12, line=dict(width=2, color='darkorange')),
            text=historical_anomalies.apply(
                lambda row: (
                    f"<b>Historical Anomaly</b><br>"
                    f"Date: {row['injection_time']}<br>"
                    f"{selected_param}: {row[selected_param]:.3f}<br>"
                    f"{color_by}: {str(row[color_col]) if pd.notnull(row[color_col]) else 'Unknown'}<br>"
                    f"Deviation: {format(float(row['anomaly_deviation']), '.3f') if pd.notnull(row['anomaly_deviation']) else '0.000'}<br>"
                    f"Severity: {'Severe' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < -0.05 else 'Moderate' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < 0 else 'Normal'}"
                ), axis=1
            ),
            hovertemplate='%{text}<extra></extra>',
            hoverlabel=dict(bgcolor='#FFF3CD', font_size=12, align='left'),
            legendgroup='historical_anomaly',
            showlegend=True
        ))

    # 3. FUTURE ANOMALIES - Predicted anomalies
    if not future_anomalies.empty and f'predicted_{selected_param}' in future_anomalies.columns:
        if 'anomaly_flag' in future_anomalies.columns:
            predicted_anomalies = future_anomalies[future_anomalies['anomaly_flag']]
        else:
            predicted_anomalies = pd.DataFrame()
        
        if not predicted_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=predicted_anomalies['predicted_date'],
                y=predicted_anomalies[f'predicted_{selected_param}'],
                mode='markers',
                name=f'Future Anomalies ({len(predicted_anomalies)})',
                marker=dict(color='red', symbol='diamond', size=12, line=dict(width=2, color='darkred')),
                text=predicted_anomalies.apply(
                    lambda row: (
                        f"<b>Future Anomaly (Predicted)</b><br>"
                        f"Date: {row['predicted_date']}<br>"
                        f"{selected_param}: {row[f'predicted_{selected_param}']:.3f}<br>"
                        f"Column: {row['column_serial_number']}<br>"
                        f"Cause: {row['anomaly_cause']}<br>"
                        f"Alert: {row.get('replacement_alert', 'N/A')}"
                    ), axis=1
                ),
                hovertemplate='%{text}<extra></extra>',
                hoverlabel=dict(bgcolor='#F8D7DA', font_size=12, align='left'),
                legendgroup='future_anomaly',
                showlegend=True
            ))
        
        # Also show all future predictions as a reference (lighter color)
        fig.add_trace(go.Scatter(
            x=future_anomalies['predicted_date'],
            y=future_anomalies[f'predicted_{selected_param}'],
            mode='markers',
            name='Future Predictions (All)',
            marker=dict(color='lightgreen', size=6, opacity=0.4),
            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>Type</b>: Predicted<extra></extra>',
            legendgroup='future',
            showlegend=True
        ))
    
    # Add threshold lines for reference
    if 'upper_threshold' in iqr_stats and 'lower_threshold' in iqr_stats:
        # Upper threshold line
        fig.add_trace(go.Scatter(
            x=[filtered_data['injection_time'].min(), 
               future_anomalies['predicted_date'].max() if not future_anomalies.empty else filtered_data['injection_time'].max()],
            y=[iqr_stats['upper_threshold'], iqr_stats['upper_threshold']],
            mode='lines',
            name='Upper Threshold',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=2, dash='dash'),
            hovertemplate=f'<b>Upper Threshold</b>: {iqr_stats["upper_threshold"]:.3f}<extra></extra>',
            showlegend=True
        ))
        
        # Lower threshold line
        fig.add_trace(go.Scatter(
            x=[filtered_data['injection_time'].min(), 
               future_anomalies['predicted_date'].max() if not future_anomalies.empty else filtered_data['injection_time'].max()],
            y=[iqr_stats['lower_threshold'], iqr_stats['lower_threshold']],
            mode='lines',
            name='Lower Threshold',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=2, dash='dash'),
            hovertemplate=f'<b>Lower Threshold</b>: {iqr_stats["lower_threshold"]:.3f}<extra></extra>',
            showlegend=True
        ))

    # Set x-axis range to include both historical and predicted dates
    x_min = filtered_data['injection_time'].min() if not filtered_data.empty else pd.Timestamp.now()
    x_max_historical = filtered_data['injection_time'].max() if not filtered_data.empty else pd.Timestamp.now()
    x_max_predicted = future_anomalies['predicted_date'].max() if not future_anomalies.empty and 'predicted_date' in future_anomalies.columns else x_max_historical
    x_max = max(x_max_historical, x_max_predicted)
    
    # Show date ranges and anomaly summary
    st.info(f"📅 **Date Ranges:** Historical: {x_min.strftime('%Y-%m-%d')} to {x_max_historical.strftime('%Y-%m-%d')} | Predictions: {x_max_predicted.strftime('%Y-%m-%d') if not future_anomalies.empty else 'N/A'}")
    
    # Show anomaly summary with colors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Historical Data Points", f"{len(filtered_data):,}", help="Total historical data points")
    with col2:
        hist_anomaly_count = len(historical_anomalies) if not historical_anomalies.empty else 0
        st.metric("Historical Anomalies", f"{hist_anomaly_count:,}", 
                  delta=f"{hist_anomaly_count/len(filtered_data)*100:.1f}%" if len(filtered_data) > 0 else "0%",
                  delta_color="inverse")
    with col3:
        if not future_anomalies.empty:
            pred_anomaly_count = future_anomalies['anomaly_flag'].sum() if 'anomaly_flag' in future_anomalies.columns else 0
            st.metric("Future Anomalies", f"{pred_anomaly_count:,}",
                      delta=f"{pred_anomaly_count/len(future_anomalies)*100:.1f}%" if len(future_anomalies) > 0 else "0%",
                      delta_color="inverse")
        else:
            st.metric("Future Anomalies", "N/A", help="No predictions loaded")
    
    # Alert status
    if not future_anomalies.empty:
        pred_anomaly_count = future_anomalies['anomaly_flag'].sum() if 'anomaly_flag' in future_anomalies.columns else 0
        if pred_anomaly_count == 0:
            st.success(f"✅ **System Status:** All {len(future_anomalies)} future predictions are NORMAL - System is stable!")
        else:
            st.warning(f"⚠️ **System Status:** {pred_anomaly_count} future anomalies detected - Review required!")
    
    fig.update_layout(
        title=dict(
            text=f'<b>{selected_param} Analysis</b><br><sub>Historical Data, Historical Anomalies & Future Predictions</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Date / Time',
        yaxis_title=f'{selected_param} Value',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        xaxis=dict(range=[x_min, x_max]),
        height=600,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Info message if predictions not available for selected parameter
    if not future_anomalies.empty and f'predicted_{selected_param}' not in future_anomalies.columns:
        available_pred_params = [col.replace('predicted_', '') for col in future_anomalies.columns if col.startswith('predicted_')]
        st.info(f"ℹ️ Predictions available for: {', '.join(available_pred_params)}. Parameter '{selected_param}' predictions not found in the file.")
    
    # Legend explanation
    st.markdown("""
    **📊 Chart Legend:**
    - 🔵 **Light Blue dots**: Historical data points (all measurements)
    - 🟠 **Orange X markers**: Historical anomalies (detected in past data)
    - 🔴 **Red Diamond markers**: Future anomalies (predicted)
    - 🟢 **Light Green dots**: Future predictions (all forecasted values)
    - 📏 **Red dashed lines**: Upper and lower anomaly thresholds
    """)

    st.subheader("Historical Data Table")
    columns_to_display = list(dict.fromkeys([
        'injection_time', selected_param, 'resolution', 'retention_time', 
        'anomaly_feature', 'anomaly_deviation'
    ]))
    st.dataframe(filtered_data[columns_to_display])

    if show_predicted_table and not future_anomalies.empty:
        st.subheader("Predicted Data Table")
        columns_to_display_pred = list(dict.fromkeys([
            'predicted_date', f'predicted_{selected_param}', 'anomaly_flag', 
            'anomaly_cause', 'replacement_alert', 'anomaly_deviation'
        ]))
        columns_to_display_pred = [col for col in columns_to_display_pred if col in future_anomalies.columns]
        st.dataframe(future_anomalies[columns_to_display_pred])

    if show_deviation_graph:
        time_range_label = deviation_time_range
        if deviation_time_range == "One Day":
            time_delta = timedelta(days=1)
        elif deviation_time_range == "One Week":
            time_delta = timedelta(days=7)
        else:
            time_delta = timedelta(days=30)

        latest_historical = filtered_data['injection_time'].max() if not filtered_data.empty else pd.Timestamp.now()
        latest_predicted = future_anomalies['predicted_date'].max() if not future_anomalies.empty else latest_historical
        latest_date = max(latest_historical, latest_predicted)

        deviation_data = filtered_data[
            (filtered_data['anomaly_deviation'] != 0) & 
            (filtered_data['injection_time'] >= latest_date - time_delta)
        ]
        pred_deviation_data = future_anomalies[
            (future_anomalies['anomaly_deviation'] != 0) & 
            (future_anomalies['predicted_date'] >= latest_date - time_delta)
        ] if not future_anomalies.empty else pd.DataFrame()

        st.subheader(f"Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)")
        
        fig_deviation = go.Figure()

        if not deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=deviation_data['injection_time'],
                y=deviation_data['anomaly_deviation'],
                mode='markers',
                name='Historical Deviations',
                marker=dict(color='blue', size=8),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Deviation</b>: %{{y:.3f}}<extra></extra>'
            ))

        if not pred_deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=pred_deviation_data['predicted_date'],
                y=pred_deviation_data['anomaly_deviation'],
                mode='markers',
                name='Predicted Deviations',
                marker=dict(color='red', size=8),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Deviation</b>: %{{y:.3f}}<extra></extra>'
            ))

        if not deviation_data.empty or not pred_deviation_data.empty:
            fig_deviation.update_layout(
                title=f'Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)',
                xaxis_title='Date',
                yaxis_title='Anomaly Deviation',
                hovermode='closest',
                showlegend=True
            )
            st.plotly_chart(fig_deviation, use_container_width=True)
        else:
            st.info(f"No deviations found for {selected_param} in the last {time_range_label}.")

    # Chromatogram Overlay Graph
    if show_chromatogram_overlay:
        st.subheader(f"Chromatogram Overlay - {selected_param} Across Injection Times")
        
        # Limit to recent injections for readability (last 50 injections)
        overlay_data = filtered_data.sort_values('injection_time').tail(50)
        
        if not overlay_data.empty and selected_param in overlay_data.columns:
            # Create color mapping based on the selected color_by field
            color_field = f"{color_by}_original" if f"{color_by}_original" in overlay_data.columns else color_by
            unique_groups = overlay_data[color_field].unique()
            
            # Create a color palette
            import plotly.express as px
            colors = px.colors.qualitative.Plotly
            color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
            
            fig_overlay = go.Figure()
            
            # Group by the color field and plot each group
            for group in unique_groups:
                group_data = overlay_data[overlay_data[color_field] == group].sort_values('injection_time')
                
                if not group_data.empty:
                    fig_overlay.add_trace(go.Scatter(
                        x=group_data['injection_time'],
                        y=group_data[selected_param],
                        mode='lines+markers',
                        name=str(group),
                        line=dict(color=color_map[group], width=2),
                        marker=dict(size=6, color=color_map[group]),
                        hovertemplate=(
                            f'<b>{color_by}</b>: {group}<br>'
                            f'<b>Injection Time</b>: %{{x}}<br>'
                            f'<b>{selected_param}</b>: %{{y:.3f}}<br>'
                            '<extra></extra>'
                        )
                    ))
            
            fig_overlay.update_layout(
                title=f'Chromatogram Overlay: {selected_param} Signal Pattern (Last 50 Injections)',
                xaxis_title='Injection Time',
                yaxis_title=f'{selected_param} Value',
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                ),
                height=500
            )
            
            st.plotly_chart(fig_overlay, use_container_width=True)
            st.info(f"📊 Displaying overlay of last 50 injections, colored by **{color_by}**. "
                   f"Total unique groups: {len(unique_groups)}")
        else:
            st.warning(f"No data available for chromatogram overlay of {selected_param}")

    st.text(f"Showing historical data for {len(filtered_data)} injections")
    st.text(f"Showing predictions for {len(future_anomalies)} future dates starting from {prediction_start.strftime('%Y-%m-%d')}")

    # --- Anomaly summary section (commented out as per user request) ---
    # if not future_anomalies.empty:
    #     load_dotenv()
    #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #
    #     anomaly_summary = future_anomalies[future_anomalies['anomaly_flag']]
    #     if not anomaly_summary.empty:
    #         summary_text = f"Summary of future anomalies for {selected_param} starting from {prediction_start.strftime('%Y-%m-%d')}:\n"
    #         for index, row in anomaly_summary.iterrows():
    #             summary_text += (
    #                 f"- Date: {row['predicted_date']}, "
    #                 f"Predicted Value: {row[f'predicted_{selected_param}']:.3f}, "
    #                 f"Cause: {row['anomaly_cause']}, "
    #                 f"Replacement Alert: {row['replacement_alert']}\n"
    #             )
    #         summary_text += "Please review these predictions and take action if necessary."
    #
    #         response = client.chat.completions.create(
    #             model="gpt-4.1-nano",
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": (
    #                         "You are a chromatography expert summarizing anomaly predictions. "
    #                         "Provide a concise summary in tabular format, including Date, Parameter, Cause, and Replacement Alert. "
    #                         "Ensure causes are specific to chromatography (e.g., column clogging for high peak width, contamination for high retention time) "
    #                         "and include actionable recommendations."
    #                         "if u dont find any data like values dont show it like NAN or anything just remove the bar and if find than only put it"
    #                     )
    #                 },
    #                 {"role": "user", "content": f"Summarize the following anomaly data: {summary_text}"}
    #             ],
    #             max_tokens=500
    #         )
    #
    #         st.subheader("Future Anomaly Summary")
    #         st.write(response.choices[0].message.content)
    #     else:
    #         st.warning("No future anomalies detected. Check model sensitivity or data range.")

if __name__ == "__main__":
    main()
# Footer
st.markdown("""
<div class="footer">
    Databricks Bundle Manager • v1.0.0 • Powered by Tetra Science
</div>
""", unsafe_allow_html=True)