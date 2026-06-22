from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout


class LSTMAutoencoderModel:
    """архитектура автоэнкодера для финансовых временных рядов."""

    @staticmethod
    def build(time_steps: int, features: int, optimizer: str = 'adam', loss: str = 'mse') -> Sequential:
        model = Sequential([
            # Входной слой
            Input(shape=(time_steps, features)),

            # --- ЭНКОДЕР ---
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),  # Исключаем 20% связей, спасая модель от переобучения

            LSTM(32, activation='tanh', return_sequences=False),
            Dropout(0.2),

            # Мост между частями сети
            RepeatVector(time_steps),

            # --- ДЕКОДЕР ---
            LSTM(32, activation='tanh', return_sequences=True),
            Dropout(0.2),

            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),

            # Выходной слой (без жесткой активации, чтобы не резать значения)
            TimeDistributed(Dense(features))
        ])

        model.compile(optimizer=optimizer, loss=loss)
        return model
