import numpy as np


class AnomalyDetector:
    """Класс для расчета ошибок восстановления и поиска аномалий."""

    def __init__(self, threshold_percentile: float = 99.0):
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def calculate_mae_loss(self, X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(X_pred - X_true), axis=1)

    def fit_threshold(self, X_true: np.ndarray, X_pred: np.ndarray) -> float:
        mae_loss = self.calculate_mae_loss(X_true, X_pred)
        self.threshold = np.percentile(mae_loss, self.threshold_percentile)
        return self.threshold

    def detect(self, X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Порог не задан. Сначала вызовите метод fit_threshold().")
        mae_loss = self.calculate_mae_loss(X_true, X_pred)
        return mae_loss > self.threshold
