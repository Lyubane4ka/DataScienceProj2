DATA_PATH = 'NASDAQ_100_Data_From_2010.csv'
TARGET_COLUMN = 'Adj Close'

# Какую конкретно акцию из датасета мы будем анализировать
TICKER = 'AAPL'

MODEL_SAVE_PATH = 'lstm_autoencoder.keras'
LOSS_PLOT_PATH = 'training_loss.png'
ANOMALY_PLOT_PATH = 'detected_anomalies.png'

TIME_STEPS = 30
FEATURES = 1
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.1
THRESHOLD_PERCENTILE = 95.0
