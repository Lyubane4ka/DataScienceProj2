import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import pandas as pd

def create_visualizations_uber(df, target_column="Close"):
    print("\n" + "=" * 50)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ АНАЛИЗА АКЦИЙ UBER")
    print("=" * 50)

    # Убедимся, что индекс является датой для корректного построения таймлайна
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # ----------------------------------------------------
    # 1. ИНТЕРАКТИВНЫЙ ГРАФИК «ЯПОНСКИЕ СВЕЧИ» (Candlestick)
    # ----------------------------------------------------
    # Проверяем наличие всех необходимых биржевых колонок
    required_cols = ["Open", "High", "Low", "Close"]
    if all(col in df.columns for col in required_cols):
        fig_candles = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Uber OHLC",
                )
            ]
        )
        fig_candles.update_layout(
            title="Интерактивный график японских свечей акций Uber",
            yaxis_title="Цена акции ($)",
            xaxis_title="Дата",
            xaxis_rangeslider_visible=True,  # Слайдер для выбора масштаба времени
            template="plotly_dark",
        )
        fig_candles.show()
        print("-> Интерактивный свечной график открыт в браузере.")
    else:
        print("-> Пропуск свечного графика: отсутствуют колонки OHLC.")

    # ----------------------------------------------------
    # 2. СТАТИЧЕСКИЕ ГРАФИКИ: ТРЕНДЫ И ОБЪЕМЫ ТОРГОВ
    # ----------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True
    )

    # График А: Динамика цены закрытия + Скользящие средние (MA)
    axes[0].plot(
        df.index,
        df[target_column],
        label=f"Цена ({target_column})",
        color="royalblue",
        linewidth=1.5,
    )

    # Краткосрочный тренд (20 дней) и долгосрочный тренд (50 дней)
    ma_20 = df[target_column].rolling(window=20).mean()
    ma_50 = df[target_column].rolling(window=50).mean()

    axes[0].plot(
        df.index, ma_20, label="MA 20 дней", color="darkorange", linestyle="--"
    )
    axes[0].plot(
        df.index, ma_50, label="MA 50 дней", color="crimson", linestyle=":"
    )

    axes[0].set_title(
        f"Анализ стоимости акций Uber (Колонка: {target_column})",
        fontsize=14,
        weight="bold",
    )
    axes[0].set_ylabel("Цена ($)", fontsize=12)
    axes[0].legend(loc="upper left")

    # График Б: Объемы торгов (Volume) с цветовым кодированием
    if "Volume" in df.columns:
        # Если цена закрытия выше открытия — красим объем в зеленый, иначе в красный
        if "Open" in df.columns:
            # Используем сочные биржевые HEX-цвета: ярко-зеленый и насыщенный красный
            color_mask = np.where(df["Close"] >= df["Open"], "#00c805", "#ff3b30")

            axes[1].bar(
                df.index,
                df["Volume"],
                color=color_mask,
                alpha=1.0,  # Максимальная яркость без прозрачности
                width=1.0,
                edgecolor="none"  # Убираем темные границы для чистоты цвета
            )
        else:
            # Яркий фиолетовый/пурпурный вместо блеклого стандартного
            axes[1].bar(
                df.index,
                df["Volume"],
                color="#9b5de5",
                alpha=1.0,
                width=1.0,
                edgecolor="none"
            )

        axes[1].set_title("Объем торгов по дням (Volume)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Количество акций", fontsize=12)
        axes[1].set_xlabel("Дата", fontsize=12)

    plt.tight_layout()
    plt.show()
    print("-> Статические графики тренда и объемов выведены на экран.")