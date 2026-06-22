import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping


class PipelineManager:
    """Класс для управления обучением, инференсом и сохранением модели."""

    def __init__(self, model: Sequential = None):
        self.model = model
        self.history = None

    def train(self, X_train: np.ndarray, epochs: int, batch_size: int, validation_split: float) -> None:
        if self.model is None:
            raise ValueError("Модель не инициализирована для обучения.")

        # Настраиваем автоматический останов при переобучении
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Следим за ошибкой на валидации
            patience=2,  # Ждем 2 эпохи ухудшения, прежде чем прервать обучение
            restore_best_weights=True  # Обязательно откатываем веса к лучшей (5-й) эпохе
        )

        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],  # Добавляем колбэк сюда
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не загружена.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        if self.model:
            self.model.save(filepath)
            print(f"[Пайплайн] Модель успешно сохранена в {filepath}")

    def load_model(self, filepath: str) -> None:
        self.model = load_model(filepath)
        print(f"[Пайплайн] Модель успешно загружена из {filepath}")
