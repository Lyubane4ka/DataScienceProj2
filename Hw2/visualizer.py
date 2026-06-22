import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesVisualizer:
    """Класс для визуализации метрик обучения и результатов детекции аномалий."""

    @staticmethod
    def plot_training_history(history, save_path: str = None) -> None:
        """Отрисовывает графики потерь (Loss) на обучении и валидации."""
        if history is None or not hasattr(history, 'history'):
            print("[Визуализатор] Нет истории обучения для отображения.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Ошибка на обучении (Train Loss)', color='teal')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Ошибка на валидации (Val Loss)', color='orange')

        plt.title('История обучения автоэнкодера (Loss History)')
        plt.xlabel('Эпохи')
        plt.ylabel('MSE (Среднеквадратичная ошибка)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Визуализатор] График обучения сохранен в {save_path}")
        plt.show()

    @staticmethod
    def plot_anomalies(index, original_series: np.ndarray, anomaly_mask: np.ndarray, save_path: str = None) -> None:
        """Отрисовывает временной ряд и накладывает красные точки на аномалии."""
        plt.figure(figsize=(15, 6))

        # Основной график цены
        plt.plot(index, original_series, label='Цена акции (Исходный ряд)', color='royalblue', alpha=0.8)

        # Выделяем точки, признанные аномальными
        anomaly_indices = np.where(anomaly_mask)[0]

        if len(anomaly_indices) > 0:
            plt.scatter(
                index[anomaly_indices],
                original_series[anomaly_indices],
                color='crimson',
                label='Обнаруженные аномалии (Сбой/Скачок)',
                zorder=5,
                s=30
            )

        plt.title('Детекция аномалий с помощью LSTM Autoencoder')
        plt.xlabel('Дата / Время')
        plt.ylabel('Значение цены')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Визуализатор] График аномалий успешно сохранен в {save_path}")
        plt.show()
