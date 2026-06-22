import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesPreprocessor:
    """Класс для предобработки данных временных рядов."""

    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.time_steps = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def create_sequences(self, data: np.ndarray, time_steps: int) -> np.ndarray:
        self.time_steps = time_steps
        sequences = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:(i + time_steps)])
        return np.array(sequences)
