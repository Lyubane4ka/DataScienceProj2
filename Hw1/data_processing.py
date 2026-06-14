import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import LabelEncoder



def processing_uber(df):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ АКЦИЙ UBER")
    print("=" * 50)

    # Проверка пропущенных значений
    missing_values = df.isnull().sum()
    print("Пропущенные значения:")
    print(missing_values[missing_values > 0])

    # Если есть пропущенные значения, заполним их
    if df.isnull().sum().sum() > 0:
        # Для числовых колонок - медианой
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Для категориальных/текстовых - модой
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    print(f"Дубликатов до очистки: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Дубликатов после очистки: {df.duplicated().sum()}")

    # Обработка выбросов для числовых колонок акций
    numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr_uber(df, col)

    # Проверяем категориальные колонки (если они есть, кроме даты)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if 'date' in categorical_cols:
        categorical_cols = categorical_cols.drop('date')  # Дату не кодируем как категорию

    print("Категориальные колонки:", categorical_cols.tolist())

    # Кодирование категориальных переменных (если добавлены сторонние фичи)
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} уникальных значений")

        if unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df = optimize_types_uber(df)
    print("Типы данных после оптимизации:")
    print(df.dtypes)

    return df


def handle_outliers_iqr_uber(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Для цен сохраняем float, для объема торгов — int
    if column == 'volume':
        target_type = 'int64'
    else:
        target_type = 'float64'

    # Ограничение выбросов
    data.loc[:, column] = data[column].clip(lower=lower_bound, upper=upper_bound).astype(target_type)

    return data


# Оптимизация типов данных для экономии памяти
def optimize_types_uber(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object' and col != 'date':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    return df


def processing_for_arima_uber(df, target_column='close'):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ARIMA (ВРЕМЕННЫЕ РЯДЫ)")
    print("=" * 50)

    # Заполнение пропусков медианой
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = df.drop_duplicates()

    # Обработка выбросов
    numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr_uber(df, col)

    # Подготовка правильного временного индекса для ARIMA
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        df.set_index('date', inplace=True)

    # Для ARIMA важна непрерывность. Настроим частоту (B - бизнес-дни, так как биржи закрыты в выходные)
    df = df.asfreq('B')

    # Заполняем пропуски в выходные дни (если они появились после смены частоты) методом forward fill
    df = df.ffill()

    df = optimize_types_uber(df)
    print("Типы данных после оптимизации:")
    print(df.dtypes)

    return df


#Функция оценивает, содержат ли цены на акции случайный тренд (unit root).p_value < 0.05:
#Если вероятность случайного блуждания близка к нулю, ряд готов к прогнозам "как есть".df[target_column].diff():
#Метод вычитает цену предыдущего дня из цены текущего. Вместо графика цены мы получаем график изменения цены,
#который колеблется вокруг стабильного математического ожидания (стационарен).
def processing_for_arima_uber(df, target_column='close'):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ARIMA (ВРЕМЕННЫЕ РЯДЫ)")
    print("=" * 50)

    # 1. Базовая очистка и заполнение пропусков
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = df.drop_duplicates()

    # 2. Обработка выбросов (используем ранее созданную handle_outliers_iqr_uber)
    numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr_uber(df, col)

    # 3. Настройка временного индекса (бизнес-дни)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        df.set_index('date', inplace=True)

    df = df.asfreq('B')  # 'B' означает бизнес-дни (без субботы и воскресенья)
    df = df.ffill()  # Заполняем пропуски предыдущими значениями

    # Оптимизация типов памяти
    df = optimize_types_uber(df)

    # 4. ТЕСТ ДИКИ-ФУЛЛЕРА НА СТАЦИОНАРНОСТЬ
    print(f"\n[Тест Дики-Фуллера] Проверка колонки: '{target_column}'")
    series_to_test = df[target_column].dropna()

    adf_result = adfuller(series_to_test)
    p_value = adf_result[1]
    print(f"ADF Статистика: {adf_result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Интерпретация p-value (критический уровень 0.05)
    if p_value < 0.05:
        print(f"-> Результат: Ряд СТАЦИОНАРЕН (p < 0.05). Можно применять параметр d=0 в ARIMA.")
    else:
        print(f"-> Результат: Ряд НЕСТАЦИОНАРЕН (p >= 0.05).")
        print("-> Автоматически создаем стационарный ряд с помощью взятия первой разности (diff)...")

        # Создаем новую колонку с первыми разностями для обучения
        diff_column_name = f"{target_column}_diff"
        df[diff_column_name] = df[target_column].diff()

        # Проверяем получившийся ряд разностей повторно
        diff_series = df[diff_column_name].dropna()
        adf_diff_result = adfuller(diff_series)

        print(f"   [После diff] Новое p-value: {adf_diff_result[1]:.4f}")
        if adf_diff_result[1] < 0.05:
            print(f"   -> Теперь ряд СТАЦИОНАРЕН. Используйте параметр d=1 в модели ARIMA.")
        else:
            print(f"   -> Внимание! Ряд все еще нестационарен. Возможно, потребуется d=2.")

    return df
