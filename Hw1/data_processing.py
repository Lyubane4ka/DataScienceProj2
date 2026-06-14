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


def optimize_types_uber_arima_lasso(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def processing_uber_arima_lasso(df):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ АКЦИЙ UBER")
    print("=" * 50)

    # Заполнение пропущенных
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(
                df[col].mode() if not df[col].mode().empty else "Unknown"
            )

    df = df.drop_duplicates()

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if "Date" in categorical_cols:
        categorical_cols = categorical_cols.drop("Date")

    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df = optimize_types_uber_arima_lasso(df)
    return df

