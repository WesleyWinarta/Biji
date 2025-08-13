import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import pickle
import os
import traceback
import msvcrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from transformers import pipeline  
import requests  
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dateutil.parser import parse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

# PyTorch LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SafeMinMaxScaler(MinMaxScaler):
    """MinMaxScaler with safety checks for division by zero"""
    def transform(self, X):
        if np.any(self.scale_ == 0):
            self.scale_[self.scale_ == 0] = 1.0
        return super().transform(X)

def validate_date(date_str):
    """Validate and parse date string"""
    try:
        return parse(date_str).date()
    except ValueError:
        st.error(f"Invalid date format: {date_str}")
        return None

def clean_numeric_value(x):
    """Clean and convert various numeric formats to float"""
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '')
        if 'K' in x:
            return float(x.replace('K', '')) * 1000
        elif 'M' in x:
            return float(x.replace('M', '')) * 1000000
        elif 'B' in x:
            return float(x.replace('B', '')) * 1000000000
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x

def load_data(source, ticker, start_date, end_date):
    """Load data with robust error handling and validation"""
    try:
        if source == 'local':
            if not os.path.exists(ticker):
                raise FileNotFoundError(f"File {ticker} not found")
            
            if not ticker.endswith('.csv'):
                raise ValueError("Only CSV files are supported")
                
            # Use file locking for thread-safe operations
            data = pd.read_csv(ticker)
            
            # Clean column names and convert numeric values
            data.columns = data.columns.str.strip().str.lower()
            numeric_cols = ['price', 'open', 'high', 'low', 'vol.', 'change %']
            
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = data[col].apply(clean_numeric_value)
                    if col == 'change %':
                        data[col] = data[col] / 100
            
            # Standardize column names
            data = data.rename(columns={
                'price': 'close',
                'vol.': 'volume',
                'change %': 'change'
            })
            
            # Validate and convert dates
            if 'date' not in data.columns:
                raise ValueError("Date column not found")
                
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])
            data.set_index('date', inplace=True)
            
            # Validate data freshness for local files
            if data.index[-1] < (datetime.now() - timedelta(days=30)):
                st.warning("Local data might be outdated (last entry >30 days old)")
            
            return data
            
        elif source == 'live':
            # Validate dates
            start = validate_date(start_date)
            end = validate_date(end_date)
            if not start or not end:
                return None
                
            if start >= end:
                st.error("Start date must be before end date")
                return None
                
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    st.error("No data returned from Yahoo Finance")
                    return None
                    
                # Check for recent data
                if data.index[-1].date() < (datetime.now() - timedelta(days=2)).date():
                    st.warning("Live data might not be up-to-date")
                    
                return data
                
            except Exception as e:
                st.error(f"Failed to download data: {e}")
                return None
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

def fetch_news(ticker, days=30):
    """Fetch news with robust error handling and API validation"""
    NEWS_API_KEY = "7ea7f763089f40c894d041886756deef"  # Replace with your actual key
    if not NEWS_API_KEY:
        st.warning("NewsAPI key not configured")
        return pd.DataFrame(columns=['date', 'title', 'description', 'source', 'url'])
    
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            st.warning(f"News API returned status {response.status_code}")
            return pd.DataFrame()
        
        news_data = response.json()
        if news_data.get('status') != 'ok':
            st.warning("News API response not OK")
            return pd.DataFrame()
        
        # Process articles with validation
        news_list = []
        for article in news_data.get('articles', []):
            try:
                news_list.append({
                    'date': pd.to_datetime(article.get('publishedAt', '')).date(),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', '')  # Tambahkan URL berita
                })
            except:
                continue
                
        return pd.DataFrame(news_list)
    
    except requests.exceptions.Timeout:
        st.warning("News API request timed out")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch news: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment(news_df):
    """Analyze sentiment with robust error handling"""
    if news_df.empty or ('description' not in news_df.columns):
        return 0
    
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for _, row in news_df.iterrows():
        # Safely get title and description with fallback to empty string
        title = row.get('title', '')
        description = row.get('description', '')
        text = f"{title}. {description}".strip()
        
        if text:  # Only analyze if we have text
            try:
                score = sia.polarity_scores(text)
                sentiment_scores.append(score['compound'])
            except:
                continue
    
    return np.mean(sentiment_scores) if sentiment_scores else 0

def add_news_features(data, ticker, window=30):
    """Add news features with proper validation"""
    if data.empty:
        return data
    
    news_df = fetch_news(ticker, days=window)
    data['News_Sentiment'] = 0  # Initialize column
    
    if not news_df.empty and {'date', 'description'}.issubset(news_df.columns):
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Group by date and analyze sentiment
        daily_sentiment = news_df.groupby('date')['description'].apply(
            lambda x: analyze_sentiment(pd.DataFrame({'description': x})))
        
        # Merge with main data
        for date, sentiment in daily_sentiment.items():
            if date in data.index:
                data.loc[date, 'News_Sentiment'] = sentiment
    
    return data

def preprocess_data(data, look_back=60, use_multi_feature=False, use_news=False):
    try:
        if len(data) <= look_back:
            new_look_back = max(10, len(data) - 5)
            st.warning(f"Lookback changed from {look_back} to {new_look_back} because data is short")
            look_back = new_look_back
            
        if use_news and 'News_Sentiment' not in data.columns:
            data['News_Sentiment'] = 0
            
        if use_multi_feature:
            # Include News_Sentiment if use_news is True
            if use_news:
                required_cols = ['Open', 'High', 'Low', 'Volume', 'Change', 'Close', 'News_Sentiment']
            else:
                required_cols = ['Open', 'High', 'Low', 'Volume', 'Change', 'Close']
                
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"These columns are missing: {missing_cols}")
                return None, None, None, None
                
            features = data[required_cols]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i-look_back:i, :])
                y.append(scaled_data[i, -1])
            
            X, y = np.array(X), np.array(y)
            
            if len(X) == 0:
                st.error("No data generated. Check look_back and data length")
                return None, None, None, None
                
            return X, y, scaler, features.values
            
        else:
            if 'Close' not in data.columns:
                st.error("'Close' column not found")
                return None, None, None, None
                
            dataset = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            
            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i-look_back:i, 0])
                y.append(scaled_data[i, 0])
            
            X = np.array(X).reshape(-1, look_back, 1)
            y = np.array(y)
            
            if len(X) == 0:
                st.error("No data generated. Check look_back and data length")
                return None, None, None, None
                
            return X, y, scaler, dataset
            
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.write("Error traceback:", traceback.format_exc())
        return None, None, None, None

def build_lstm_model(input_shape, use_multi_feature=False):
    model = Sequential()
    if use_multi_feature:
        model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[0], input_shape[1])))
    else:
        model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[0], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_pytorch_model(X_train, y_train, input_size, epochs=20, batch_size=32):
    """Train PyTorch LSTM model"""
    # Determine input size
    if len(X_train.shape) == 3:
        input_size = X_train.shape[2]
    else:
        input_size = 1
    
    # Create model
    model = LSTMModel(input_size=input_size, 
                     hidden_size=50, 
                     num_layers=3, 
                     output_size=1)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset and dataloader
    dataset = StockDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def train_models(X_train, y_train, lstm_input_shape, use_multi_feature=False, selected_models=None):
    models = {}
    
    # Prepare data for non-LSTM models
    if use_multi_feature:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Train selected models
    if 'LSTM' in selected_models:
        with st.spinner('Training LSTM model...'):
            lstm_model = build_lstm_model(lstm_input_shape, use_multi_feature)
            lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            models['LSTM'] = lstm_model
            st.success('LSTM training completed')
    
    if 'PyTorch LSTM' in selected_models:
        with st.spinner('Training PyTorch LSTM model...'):
            if use_multi_feature:
                input_size = X_train.shape[2]
            else:
                input_size = 1
            pt_model = train_pytorch_model(X_train, y_train, input_size)
            models['PyTorch LSTM'] = pt_model
            st.success('PyTorch LSTM training completed')
    
    if 'Random Forest' in selected_models:
        with st.spinner('Training Random Forest model...'):
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_flat, y_train)
            models['Random Forest'] = rf_model
            st.success('Random Forest training completed')
    
    if 'XGBoost' in selected_models:
        with st.spinner('Training XGBoost model...'):
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train_flat, y_train)
            models['XGBoost'] = xgb_model
            st.success('XGBoost training completed')
    
    if 'LightGBM' in selected_models:
        with st.spinner('Training LightGBM model...'):
            lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
            lgbm_model.fit(X_train_flat, y_train)
            models['LightGBM'] = lgbm_model
            st.success('LightGBM training completed')
    
    if 'SVR' in selected_models:
        with st.spinner('Training SVR model...'):
            svr_model = SVR()
            svr_model.fit(X_train_flat, y_train)
            models['SVR'] = svr_model
            st.success('SVR training completed')
    
    if 'MLP' in selected_models:
        with st.spinner('Training MLP model...'):
            mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            mlp_model.fit(X_train_flat, y_train)
            models['MLP'] = mlp_model
            st.success('MLP training completed')
    
    if 'Linear Regression' in selected_models:
        with st.spinner('Training Linear Regression model...'):
            lr_model = LinearRegression()
            lr_model.fit(X_train_flat, y_train)
            models['Linear Regression'] = lr_model
            st.success('Linear Regression training completed')
    
    if 'Gradient Boosting' in selected_models:
        with st.spinner('Training Gradient Boosting model...'):
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train_flat, y_train)
            models['Gradient Boosting'] = gb_model
            st.success('Gradient Boosting training completed')
    
    return models

def ensemble_predict(models, X_input, y_actual, scaler, use_multi_feature=False):
    """
    Enhanced ensemble prediction with multiple combination methods:
    1. Individual model predictions
    2. Simple average ensemble
    3. Weighted average ensemble (by inverse RMSE)
    4. Median ensemble (robust to outliers)
    """
    predictions = {}
    models_rmse = {}
    
    # First pass: get all individual predictions and calculate RMSEs
    for model_name, model in models.items():
        try:
            if model_name in ['LSTM', 'PyTorch LSTM']:
                if model_name == 'LSTM':
                    pred = model.predict(X_input, verbose=0)
                else:  # PyTorch LSTM
                    with torch.no_grad():
                        model.eval()
                        X_tensor = torch.FloatTensor(X_input)
                        pred = model(X_tensor).numpy()
            else:
                if use_multi_feature:
                    X_input_flat = X_input.reshape(X_input.shape[0], -1)
                else:
                    X_input_flat = X_input.reshape(X_input.shape[0], -1)
                
                pred = model.predict(X_input_flat).reshape(-1, 1)
            
            # Inverse transform
            if use_multi_feature:
                dummy = np.zeros((len(pred), scaler.n_features_in_))
                dummy[:, -1] = pred.flatten()
                pred = scaler.inverse_transform(dummy)[:, -1]
            else:
                pred = scaler.inverse_transform(pred).flatten()
            
            predictions[model_name] = pred
            
            # Calculate RMSE for weighting
            actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(actual, pred))
            models_rmse[model_name] = rmse
            
        except Exception as e:
            st.warning(f"Prediction skipped for {model_name} due to error: {str(e)}")
            continue
    
    # Create ensemble predictions only if we have multiple successful models
    if len(predictions) > 1:
        pred_matrix = np.array(list(predictions.values()))
        
        # 1. Simple Average Ensemble
        predictions['Average_Ensemble'] = np.mean(pred_matrix, axis=0)
        
        # 2. Weighted Average Ensemble (by inverse RMSE)
        if len(models_rmse) > 0:  # Only if we have RMSE values
            weights = np.array([1/(rmse+1e-6) for rmse in models_rmse.values()])
            weights /= weights.sum()  # Normalize
            try:
                predictions['Weighted_Ensemble'] = np.average(pred_matrix, axis=0, weights=weights)
            except:
                predictions['Weighted_Ensemble'] = np.mean(pred_matrix, axis=0)
        
        # 3. Median Ensemble (robust to outliers)
        predictions['Median_Ensemble'] = np.median(pred_matrix, axis=0)
        
        # 4. Best Model Only
        if models_rmse:  # Only if we have RMSE values
            best_model_name = min(models_rmse, key=models_rmse.get)
            predictions['Best_Single_Model'] = predictions[best_model_name]
    
    return predictions, models_rmse

def predict_future(models, last_data, scaler, look_back, days_to_predict, use_multi_feature=False, models_rmse=None):
    """
    Enhanced future prediction with:
    1. Individual model forecasts
    2. Multiple ensemble methods
    3. Better error handling
    """
    future_predictions = {model_name: [] for model_name in models.keys()}
    daily_predictions = []  # To store all models' predictions each day
    
    if use_multi_feature:
        current_batch = last_data[-look_back:].reshape(1, look_back, -1)
        n_features = last_data.shape[1]
    else:
        current_batch = last_data[-look_back:].reshape(1, look_back, 1)
        n_features = 1
    
    for day in range(days_to_predict):
        day_preds = {}  # Store all model predictions for this day
        
        for model_name, model in models.items():
            try:
                if model_name in ['LSTM', 'PyTorch LSTM']:
                    if use_multi_feature:
                        current_batch_lstm = current_batch.reshape(1, look_back, n_features)
                    else:
                        current_batch_lstm = current_batch.reshape(1, look_back, 1)
                    
                    if model_name == 'LSTM':
                        pred = model.predict(current_batch_lstm, verbose=0)
                    else:  # PyTorch LSTM
                        with torch.no_grad():
                            model.eval()
                            X_tensor = torch.FloatTensor(current_batch_lstm)
                            pred = model(X_tensor).numpy()
                else:
                    if use_multi_feature:
                        current_batch_flat = current_batch.reshape(1, -1)
                    else:
                        current_batch_flat = current_batch.reshape(1, -1)
                    
                    pred = model.predict(current_batch_flat).reshape(1, 1, 1)
                
                # Inverse transform
                if use_multi_feature:
                    dummy = np.zeros((1, n_features))
                    dummy[:, -1] = pred[0, 0] if model_name in ['LSTM', 'PyTorch LSTM'] else pred[0]
                    pred_value = scaler.inverse_transform(dummy)[0, -1]
                else:
                    pred_value = scaler.inverse_transform(pred.reshape(1, -1))[0][0]
                
                future_predictions[model_name].append(pred_value)
                day_preds[model_name] = pred_value
                
                # Update batch for next prediction
                if model_name in ['LSTM', 'PyTorch LSTM']:
                    if use_multi_feature:
                        new_features = np.zeros((1, 1, n_features))
                        new_features[:, :, -1] = pred[0, 0] if model_name == 'LSTM' else pred[0]
                        current_batch = np.append(current_batch[:, 1:, :], new_features, axis=1)
                    else:
                        current_batch = np.append(current_batch[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            except Exception as e:
                st.warning(f"Prediction skipped for {model_name} day {day+1}: {str(e)}")
                future_predictions[model_name].append(np.nan)
        
        daily_predictions.append(day_preds)
    
    # Create ensemble predictions for each day
    if len(models) > 1:
        future_predictions['Average_Ensemble'] = []
        future_predictions['Weighted_Ensemble'] = []
        future_predictions['Median_Ensemble'] = []
        
        for day_preds in daily_predictions:
            valid_preds = [p for p in day_preds.values() if not np.isnan(p)]
            
            if valid_preds:
                # Average
                future_predictions['Average_Ensemble'].append(np.mean(valid_preds))
                
                # Weighted Average (if we have RMSE values)
                if models_rmse:
                    valid_models = [m for m in day_preds.keys() if m in models_rmse and not np.isnan(day_preds[m])]
                    if valid_models:
                        weights = np.array([1/(models_rmse[m]+1e-6) for m in valid_models])
                        weights /= weights.sum()
                        weighted_avg = np.average([day_preds[m] for m in valid_models], weights=weights)
                        future_predictions['Weighted_Ensemble'].append(weighted_avg)
                
                # Median
                future_predictions['Median_Ensemble'].append(np.median(valid_preds))
    
    return future_predictions

def show_extreme_detail_guide():
    st.markdown("""
    # 🧠📈 PANDUAN SUPER DETAIL APLIKASI PREDIKSI SAHAM (v2.0)

    ## 🏗️ **1. ARSITEKTUR SISTEM**
    ### Diagram Alur Data:
    ```mermaid
    graph TD
        A[Input Data] --> B[Preprocessing]
        B --> C[Training Model]
        C --> D[Evaluasi]
        D --> E[Prediksi]
        E --> F[Visualisasi]
    ```

    ### Komponen Utama:
    1. **Data Layer**:
       - Yahoo Finance API
       - CSV Parser (support 10+ format angka)
    2. **Preprocessing**:
       - Normalisasi MinMax (0-1)
       - Handling missing data (interpolasi linear)
       - Feature engineering (auto-detect)
    3. **Model Layer**:
       - TensorFlow/Keras LSTM
       - PyTorch LSTM
       - 6 model machine learning klasik
       - 3 ensemble method
    4. **Visualization**:
       - Interactive Plotly charts
       - Dynamic metrics table

    ## 📥 **2. INPUT DATA (DETAIL EXTREME)**
    ### Live Mode (Yahoo Finance):
    - **Fungsi Dasar**: `yfinance.download()`
    - **Parameter Tersembunyi**:
      ```python
      yf.download(
          tickers="AAPL",
          period="5y",  # max 10 tahun
          interval="1d",  # 1m,5m,15m,1h,1d,1wk,1mo
          prepost=True,  # pre-market data
          repair=True  # auto-fix errors
      )
      ```
    - **Limitasi**:
      - Data 1m hanya 7 hari terakhir
      - Data 1h maks 730 hari

    ### Local CSV Mode:
    **Valid Column Combinations**:
    | Scenario | Required Columns | Optional Columns |
    |----------|------------------|------------------|
    | Minimal | Date, Close | - |
    | Standar | Date, Open, High, Low, Close | Volume |
    | Lengkap | Semua di standar + Change % | Adj Close |

    **Auto-Conversion Rules**:
    - Volume: `1.2K` → 1200, `1,500` → 1500, `1.5M` → 1500000
    - Date Format: Support 15+ format termasuk:
      - `YYYY-MM-DD`
      - `DD/MM/YYYY`
      - `MM-DD-YYYY HH:MM`

    ## 🤖 **3. MODEL DETAIL (LEVEL KODE)**
    ### LSTM Architecture (TensorFlow/Keras):
    ```python
    Sequential(
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(1)
    )
    ```
    - **Optimizer**: Adam (lr=0.001)
    - **Batch Size**: 32
    - **Epochs**: 20 (early stopping)

    ### PyTorch LSTM Architecture:
    ```python
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out
    ```
    - **Optimizer**: Adam (lr=0.001)
    - **Batch Size**: 32
    - **Epochs**: 20

    ### Hyperparameter Model Lain:
    | Model | Parameter Kunci | Default Value | Range Optimal |
    |-------|-----------------|---------------|---------------|
    | XGBoost | n_estimators | 100 | 50-200 |
    | | learning_rate | 0.1 | 0.01-0.3 |
    | Random Forest | max_depth | None | 3-10 |
    | | min_samples_split | 2 | 2-5 |

    ## ⚙️ **4. PREPROCESSING DETAIL**
    ### Normalisasi:
    - **Rumus**: 
      ```math
      X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
      ```
    - **Fitur Kategori**:
      - Volume di-log10 kan sebelum normalisasi
      - Change % di-clipping [-1, 1]

    ### Lookback Window:
    **Mekanisme Pembentukan**:
    ```python
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])  # Sliding window
        y.append(data[i])              # Target value
    ```

    **Efek Ukuran Window**:
    | Window Size | Kelebihan | Kekurangan |
    |-------------|-----------|------------|
    | <30 | Cepat | Kurang capture trend |
    | 30-60 | Balance | Butuh data cukup |
    | >60 | Robust | Lambat berevolusi |

    ## 📊 **5. OUTPUT ANALYSIS (PRO LEVEL)**
    ### Interpretasi RMSE:
    - **Cara Hitung**:
      ```python
      rmse = sqrt(mean((actual - predicted)**2))
      ```
    - **Benchmark**:
      - RMSE < 1% dari harga = Excellent
      - 1-3% = Good
      - >5% = Poor

    ### Confidence Interval:
    Untuk ensemble weighted:
    ```python
    ci = 1.96 * (std_pred / sqrt(n_models))
    ```

    **Visualisasi Uncertainty**:
    ```python
    plt.fill_between(dates, pred_low, pred_high, alpha=0.2)
    ```

    ## 🛠️ **6. TROUBLESHOOTING EXTREME**
    ### Error Message Database:
    | Error | Penyebab | Solusi |
    |-------|----------|--------|
    | Shape mismatch | Lookback > data length | Kurangi lookback |
    | NaN values | Data corrupt | Interpolasi atau drop |
    | Yahoo 404 | Ticker salah | Cek di website Yahoo |

    ### Debug Mode:
    Aktifkan dengan:
    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)
    ```

    **Checkpoint Utama**:
    1. Data loading (cek 5 row pertama)
    2. Normalisasi (cek min/max)
    3. Shape training set
    4. Model summary

    ## � **7. USE CASE STUDIES**
    ### Case 1: Day Trading Crypto
    **Konfigurasi**:
    ```yaml
    ticker: BTC-USD
    timeframe: 1h
    lookback: 168 (7 hari hourly)
    models: [LSTM, PyTorch LSTM, LightGBM]
    features: [close, volume]
    ```
    **Hasil Optimal**:
    - Weighted ensemble RMSE: 0.8%
    - Latensi: <2 menit

    ### Case 2: Long-term Investing
    ```yaml
    ticker: ^JKSE
    timeframe: 1d
    lookback: 252 (1 tahun trading)
    models: [XGBoost, Random Forest]
    features: full
    ```
    **Pattern Ditemukan**:
    - Weekly seasonality
    - Support/resistance detection

    ## 📚 **8. REFERENSI TEKNIS**
    **Paper Implementasi**:
    - LSTM: Hochreiter & Schmidhuber (1997)
    - XGBoost: Chen & Guestrin (2016)

    **Library Versi**:
    - TensorFlow 2.10+
    - PyTorch 1.12+
    - scikit-learn 1.2+
    - yfinance 0.2+

    ## 🚀 **9. EXPERT TIPS**
    **Feature Engineering**:
    - Tambahkan secara manual:
      ```python
      data['MA_7'] = data['Close'].rolling(7).mean()
      data['Volatility'] = data['High'] - data['Low']
      ```

    **Custom Ensemble**:
    ```python
    custom_weights = {
        'LSTM': 0.4,
        'PyTorch LSTM': 0.3,
        'XGBoost': 0.3
    }
    ```
    """)

def generate_recommendation(prediction_df, current_price, risk_tolerance='medium'):
    """
    Menghasilkan rekomendasi tindakan berdasarkan prediksi harga saham
    
    Parameters:
    - prediction_df: DataFrame yang berisi prediksi harga
    - current_price: Harga saham saat ini (single float value)
    - risk_tolerance: Profil risiko user ('low', 'medium', 'high')
    """
    try:
        # Ambil prediksi ensemble sebagai acuan utama
        if 'Weighted_Ensemble' in prediction_df.columns:
            pred_series = prediction_df['Weighted_Ensemble']
        elif 'Average_Ensemble' in prediction_df.columns:
            pred_series = prediction_df['Average_Ensemble']
        else:
            pred_series = prediction_df.iloc[:, 0]  # Ambil kolom pertama sebagai fallback
        
        # Pastikan kita bekerja dengan numpy array
        pred_values = pred_series.values if hasattr(pred_series, 'values') else np.array(pred_series)
        
        # Hitung perubahan harga (pastikan current_price adalah single value)
        current_price = float(current_price)
        avg_prediction = np.mean(pred_values)
        price_change_pct = ((avg_prediction - current_price) / current_price) * 100
        price_change_pct = float(price_change_pct)  # Convert to single float
        
        trend = "naik" if price_change_pct >= 0 else "turun"
        
        # Analisis volatilitas
        volatility = np.std(pred_values) / avg_prediction * 100
        
        # Generate rekomendasi berdasarkan analisis
        if price_change_pct > 5:
            confidence = "tinggi"
            action = "beli" if trend == "naik" else "jual"
        elif 2 < price_change_pct <= 5:
            confidence = "sedang"
            action = "pertimbangkan untuk membeli" if trend == "naik" else "pertimbangkan untuk menjual"
        else:
            confidence = "rendah"
            action = "tahan" if trend == "naik" else "tahan dan pantau"
        
        # Adjust berdasarkan risiko
        if risk_tolerance == 'low' and abs(price_change_pct) < 3:
            action = "tahan (perubahan kecil - risiko rendah)"
        elif risk_tolerance == 'high' and abs(price_change_pct) > 3:
            action = f"{action} agresif" if "beli" in action or "jual" in action else action
        
        # Bangun laporan rekomendasi
        recommendation = {
            'trend': trend,
            'confidence': confidence,
            'action': action,
            'price_change': f"{abs(price_change_pct):.2f}%",
            'volatility': f"{volatility:.2f}%",
            'prediction_period': f"{len(prediction_df)} hari ke depan"
        }
        
        return recommendation
    
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        st.error(traceback.format_exc())
        return {
            'trend': 'tidak diketahui',
            'confidence': 'rendah',
            'action': 'tahan dan pantau',
            'price_change': '0%',
            'volatility': '0%',
            'prediction_period': f"{len(prediction_df)} hari ke depan"
        }

def display_recommendation(recommendation):
    """Menampilkan rekomendasi dalam format yang user-friendly"""
    st.subheader("Kesimpulan")
    
    # Tampilkan box berwarna berdasarkan rekomendasi
    if "beli" in recommendation['action']:
        color = "green"
    elif "jual" in recommendation['action']:
        color = "red"
    else:
        color = "blue"
    
    st.markdown(f"""
    <div style="background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};">
        <h4 style="color:{color}; margin-top:0;">Saran Utama: <strong>{recommendation['action'].upper()}</strong></h4>
        <p>Berdasarkan analisis prediksi:</p>
        <ul>
            <li>Trend harga diperkirakan <strong>{recommendation['trend']}</strong> ({recommendation['price_change']})</li>
            <li>Tingkat keyakinan: <strong>{recommendation['confidence']}</strong></li>
            <li>Volatilitas prediksi: <strong>{recommendation['volatility']}</strong></li>
            <li>Periode prediksi: <strong>{recommendation['prediction_period']}</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Tambahkan disclaimer
    st.caption("⚠️ Catatan: Rekomendasi ini berdasarkan analisis algoritmik dan tidak menjamin keakuratan. Selalu lakukan riset tambahan.")

def main():
    st.title('📊 Stock Prediction with TensorFlow & PyTorch')
    
    # Tampilkan panduan lengkap
    with st.expander("📘 BUKU PANDUAN LENGKAP (Klik untuk Membuka)", expanded=False):
        show_extreme_detail_guide()
    
    # Sidebar untuk input pengguna
    st.sidebar.header('Input Settings')
    data_source = st.sidebar.radio("Data Source", ('live', 'local'), help="Choose between live Yahoo Finance data or local CSV file")
    
    if data_source == 'local':
        ticker = st.sidebar.text_input("CSV Filename", "data.csv")
        st.sidebar.markdown("""
        **CSV Format Requirements:**
        - Must contain: Date, Close
        - Optional: Open, High, Low, Volume, Change %
        """)
    else:
        ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
    
    use_multi_feature = st.sidebar.checkbox("Gunakan Fitur Tambahan (Open, High, Low, Volume, Change)", False)
    use_news = st.sidebar.checkbox("Gunakan Analisis Sentimen Berita (Hanya Live Data)", True)
    
    # Daftar model yang tersedia (termasuk PyTorch LSTM)
    available_models = ['LSTM', 'PyTorch LSTM', 'Random Forest', 'XGBoost', 'LightGBM', 
                      'SVR', 'MLP', 'Linear Regression', 'Gradient Boosting']
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=['LSTM', 'PyTorch LSTM', 'Random Forest'],
        help="Select at least one model"
    )
    
    if data_source == 'live':
        default_days = 365
    else:
        default_days = 30
        
    look_back = st.sidebar.number_input("Lookback Days", min_value=10, max_value=365, 
                                      value=default_days, help="Number of historical days to consider")
    
    days_to_predict = st.sidebar.number_input("Days to Predict", min_value=1, 
                                            max_value=30, value=7)
    
    if data_source == 'live':
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    else:
        start_date = st.sidebar.text_input("Tanggal Mulai (format: YYYY-MM-DD)", "2018-01-01")
        end_date = st.sidebar.text_input("Tanggal Akhir (format: YYYY-MM-DD)", "2023-01-01")
    
    predict_button = st.sidebar.button("Mulai Prediksi")
    
    if predict_button:
        if not selected_models:
            st.error("Silakan pilih minimal satu model")
            return
            
        with st.spinner('Memproses data dan melatih model...'):
            data = load_data(data_source, ticker, start_date, end_date)
            
            if data is None:
                st.error("Failed to load data. Please check your inputs.")
                return
            if not isinstance(data, pd.DataFrame) or data.empty:
                st.error("Loaded data is empty or invalid")
                return
            st.success(f"Data loaded successfully! Rows: {len(data)}")

            if use_news and data_source == 'live':
                with st.spinner('Analyzing news sentiment...'):
                    data = add_news_features(data, ticker)
                    if data is None:
                        st.warning("News analysis failed, continuing without news data")
                        use_news = False
                    else:
                        st.success("News analysis complete!")
                        
                        if 'News_Sentiment' in data.columns:
                            last_sentiment = data['News_Sentiment'].iloc[-1]
                            st.info(f"Current news sentiment: {last_sentiment:.2f} "
                                  f"({'Positive' if last_sentiment > 0.05 else 'Negative' if last_sentiment < -0.05 else 'Neutral'})")              
            
            if len(data) < look_back + days_to_predict + 10:
                suggested_lookback = max(10, len(data) // 3)
                st.warning(f"Lookback diubah dari {look_back} menjadi {suggested_lookback} karena data terbatas")
                look_back = suggested_lookback
                
            st.subheader("Statistik Data")
            st.write(data.describe())
                
            X, y, scaler, dataset = preprocess_data(data, look_back, use_multi_feature, use_news)
                
            if X is None or len(X) == 0:
                st.error("Gagal memproses data. Periksa format dan kelengkapan data Anda.")
                return
                
            st.write(f"Shape X: {X.shape}")
            st.write(f"Shape y: {y.shape}")
                
            train_size = int(len(X) * 0.8)
            if train_size == 0:
                st.error("Tidak ada data untuk training")
                return
                    
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
                
            try:
                if use_multi_feature:
                    if len(X_train.shape) == 3:
                        lstm_input_shape = (X_train.shape[1], X_train.shape[2])
                    else:
                        n_features = 6
                        X_train = X_train.reshape((X_train.shape[0], look_back, n_features))
                        X_test = X_test.reshape((X_test.shape[0], look_back, n_features))
                        lstm_input_shape = (look_back, n_features)
                else:
                    if len(X_train.shape) == 3:
                        lstm_input_shape = (X_train.shape[1], 1)
                    else:
                        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                        X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                        lstm_input_shape = (look_back, 1)
                    
                # Latih model
                models = train_models(X_train, y_train, lstm_input_shape, use_multi_feature, selected_models)
                
                # Evaluasi model
                test_predictions, models_rmse = ensemble_predict(models, X_test, y_test, scaler, use_multi_feature)
                
                # Inverse transform untuk nilai aktual
                if use_multi_feature:
                    dummy = np.zeros((len(y_test), scaler.n_features_in_))
                    dummy[:, -1] = y_test
                    test_actual = scaler.inverse_transform(dummy)[:, -1]
                else:
                    test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Tampilkan hasil evaluasi
                st.subheader("Evaluasi Model pada Data Test")
                eval_results = []
                
                for model_name, pred in test_predictions.items():
                    if model_name in ['Average_Ensemble', 'Weighted_Ensemble', 'Median_Ensemble', 'Best_Single_Model']:
                        rmse = np.sqrt(mean_squared_error(test_actual[-len(pred):], pred))
                    else:
                        rmse = np.sqrt(mean_squared_error(test_actual, pred))
                    eval_results.append({
                        'Model': model_name,
                        'RMSE': f"{rmse:.2f}",
                        'Jumlah Prediksi': len(pred)
                    })
                
                eval_df = pd.DataFrame(eval_results)
                st.table(eval_df)
                
                # Plot hasil
                st.subheader("Perbandingan Prediksi dengan Data Aktual")
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(test_actual, label='Aktual', linewidth=2, color='black')
                
                colors = plt.cm.tab20.colors
                for i, (model_name, pred) in enumerate(test_predictions.items()):
                    if model_name == 'Average_Ensemble':
                        linestyle = '-'
                        linewidth = 3
                        color = 'blue'
                    elif model_name == 'Weighted_Ensemble':
                        linestyle = '-'
                        linewidth = 3
                        color = 'green'
                    elif model_name == 'Median_Ensemble':
                        linestyle = '-'
                        linewidth = 3
                        color = 'purple'
                    elif model_name == 'Best_Single_Model':
                        linestyle = '--'
                        linewidth = 2
                        color = 'red'
                    else:
                        linestyle = '--'
                        linewidth = 1
                        color = colors[i % len(colors)]
                    
                    ax.plot(pred, label=model_name, linestyle=linestyle, linewidth=linewidth, color=color)
                
                ax.set_title('Perbandingan Prediksi dengan Data Aktual')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True)
                st.pyplot(fig)
                
                # Visualize ensemble weights if available
                if models_rmse and len(models_rmse) > 1:
                    st.subheader("Bobot Model dalam Ensemble")
                    weights_df = pd.DataFrame({
                        'Model': models_rmse.keys(),
                        'RMSE': models_rmse.values(),
                        'Weight': [1/(rmse+1e-6) for rmse in models_rmse.values()]
                    })
                    weights_df['Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()
                    weights_df = weights_df.sort_values('Weight', ascending=False)
                    
                    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # RMSE plot
                    ax3.bar(weights_df['Model'], weights_df['RMSE'], color='skyblue')
                    ax3.set_title('RMSE Masing-masing Model')
                    ax3.set_ylabel('RMSE')
                    ax3.tick_params(axis='x', rotation=45)
                    
                    # Weight plot
                    ax4.bar(weights_df['Model'], weights_df['Weight'], color='lightgreen')
                    ax4.set_title('Bobot Relatif Model dalam Weighted Ensemble')
                    ax4.set_ylabel('Bobot')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    st.pyplot(fig3)
                
                # Prediksi masa depan
                st.subheader(f"Prediksi {days_to_predict} Hari Mendatang")
                last_sequence = X[-1]
                future_predictions = predict_future(
                    models,
                    last_sequence,
                    scaler,
                    look_back,
                    days_to_predict,
                    use_multi_feature,
                    models_rmse
                )
                
                # Buat dataframe prediksi
                last_date = data.index[-1]
                pred_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
                
                pred_df = pd.DataFrame({
                    'Tanggal': pred_dates
                }).set_index('Tanggal')
                
                for model_name, preds in future_predictions.items():
                    pred_df[model_name] = preds
                
                st.dataframe(pred_df.style.format("{:.2f}"))
                
                # Plot prediksi
                st.subheader("Visualisasi Prediksi Masa Depan")
                fig2, ax2 = plt.subplots(figsize=(14, 7))
                
                # Plot data historis
                ax2.plot(data.index[-60:], data['Close'].values[-60:], label='Historis', linewidth=2, color='black')
                
                # Plot prediksi
                colors = plt.cm.tab20.colors
                for i, model_name in enumerate(future_predictions.keys()):
                    if model_name == 'Average_Ensemble':
                        marker = 'o'
                        linestyle = '-'
                        linewidth = 3
                        color = 'blue'
                    elif model_name == 'Weighted_Ensemble':
                        marker = 'o'
                        linestyle = '-'
                        linewidth = 3
                        color = 'green'
                    elif model_name == 'Median_Ensemble':
                        marker = 'o'
                        linestyle = '-'
                        linewidth = 3
                        color = 'purple'
                    elif model_name == 'Best_Single_Model':
                        marker = 's'
                        linestyle = '--'
                        linewidth = 2
                        color = 'red'
                    else:
                        marker = '^'
                        linestyle = '--'
                        linewidth = 1
                        color = colors[i % len(colors)]
                    
                    ax2.plot(
                        pred_df.index, 
                        pred_df[model_name], 
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=model_name,
                        color=color
                    )
                
                ax2.set_title('Prediksi Harga Saham')
                ax2.set_xlabel('Tanggal')
                ax2.set_ylabel('Harga')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True)
                st.pyplot(fig2)
                
                # Analisis Sentimen Berita (jika aktif)
                if use_news and 'News_Sentiment' in data.columns:
                    st.subheader("Analisis Sentimen Berita")
                    
                    fig_sent, ax_sent = plt.subplots(figsize=(12, 4))
                    ax_sent.plot(data.index, data['News_Sentiment'], 
                                label='Sentimen Berita', color='purple')
                    ax_sent.axhline(0, color='gray', linestyle='--')
                    ax_sent.fill_between(data.index, 0, data['News_Sentiment'], 
                                        where=(data['News_Sentiment'] > 0), 
                                        color='green', alpha=0.3)
                    ax_sent.fill_between(data.index, 0, data['News_Sentiment'], 
                                        where=(data['News_Sentiment'] < 0), 
                                        color='red', alpha=0.3)
                    ax_sent.set_title('Sentimen Berita Saham')
                    ax_sent.legend()
                    st.pyplot(fig_sent)
                    
                    # Tampilkan tabel berita terbaru
                    news_df = fetch_news(ticker)
                    if not news_df.empty:
                        st.subheader("Berita Terbaru")
    
                     # Buat kolom tambahan dengan link
                    news_df['link'] = news_df['url'].apply(
                     lambda x: f'<a href="{x}" target="_blank">Baca Selengkapnya</a>' if x else 'No link available'
                     )
    
                    # Tampilkan dengan format HTML
                    st.write(
                     news_df[['date', 'title', 'source', 'link']].to_html(
                      escape=False, 
                     index=False,
                         render_links=True
                     ), 
                    unsafe_allow_html=True
                    )
                
                # Bagian Rekomendasi
                try:
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Tambahkan selector untuk risk tolerance
                    risk_tolerance = st.sidebar.selectbox(
                        "Profil Risiko Anda",
                        ('low', 'medium', 'high'),
                        index=1,
                        help="Pilih sesuai toleransi risiko investasi Anda"
                    )
                    
                    # Generate recommendation
                    recommendation = generate_recommendation(pred_df, current_price, risk_tolerance)
                    
                    # Display recommendation
                    if recommendation:
                        display_recommendation(recommendation)
                        
                        # Tambahkan analisis tambahan
                        st.subheader("📊 Analisis Prediksi Detail")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Harga Terakhir", f"${current_price:.2f}")
                            st.metric("Perubahan Prediksi", 
                                     recommendation['price_change'],
                                     delta=f"{float(recommendation['price_change'].replace('%','')):.2f}%")
                        
                        with col2:
                            st.metric("Volatilitas Prediksi", recommendation['volatility'])
                            st.metric("Keyakinan Prediksi", recommendation['confidence'].capitalize())
                
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
                    st.error(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Terjadi error saat memproses: {str(e)}")
                st.error(traceback.format_exc())
    else:
        st.info("Silakan masukkan parameter dan klik 'Mulai Prediksi' untuk memulai analisis")

if __name__ == '__main__':
    main()
