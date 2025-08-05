@echo off
REM run.bat - Script untuk menjalankan aplikasi prediksi saham di Windows

echo Memeriksa dan menginstal dependensi...
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance streamlit requests nltk xgboost lightgbm transformers python-dateutil

echo Mengunduh data lexicon untuk analisis sentimen...
python -c "import nltk; nltk.download('vader_lexicon')"

echo Menjalankan aplikasi prediksi saham...
python -m streamlit run prediksi_saham.py

pause