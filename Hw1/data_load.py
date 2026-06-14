import pandas as pd
import sqlite3


def create_table_in_db_uber():
    # 1. Подключение к базе данных акций Uber
    conn = sqlite3.connect("uber_stocks.db")
    cursor = conn.cursor()

    # 2. Создание таблицы stocks с правильными типами данных
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            adj_close REAL NOT NULL,
            volume INTEGER NOT NULL
        )
    """
    )

    # 3. Чтение исходного CSV-файла (замените путь на ваш актуальный)
    # Используйте правильное имя файла из датасета Kaggle, например: 'uber_stocks.csv'
    df = pd.read_csv("../Hw1/uber_stock_data.csv")

    # 4. Импорт данных из DataFrame в SQLite
    # if_exists='replace' автоматически перезапишет таблицу, если она уже была
    df.to_sql("stocks", conn, if_exists="replace", index=False)

    # 5. Проверка первых 10 записей из созданной таблицы БД
    query = "SELECT * FROM stocks LIMIT 10;"
    print("Первые 10 записей из базы данных SQLite:")
    print(pd.read_sql_query(query, conn))

    # 6. Вывод технической информации о DataFrame
    print("\nDataFrame Info:")
    print(df.info())

    # 7. Вывод базовой описательной статистики (mean, min, max и т.д.)
    print("\nSummary statistics:")
    print(df.describe())

    # 8. Сохранение изменений и закрытие соединения
    conn.commit()
    conn.close()
    print("\nБаза данных успешно создана и заполнена данными!")


# Загрузка данных из SQLite
def load_data_from_db():
    conn = sqlite3.connect('uber_stocks.db')
    df = pd.read_sql('SELECT * FROM stocks', conn)
    conn.close()
    return df


def describe_uber_stocks():
    # Подключение к базе данных акций Uber
    conn = sqlite3.connect("uber_stocks.db")

    # Проверка структуры таблицы
    query_structure = "PRAGMA table_info(stocks);"
    structure = pd.read_sql_query(query_structure, conn)
    print("Структура таблицы stocks:")
    print(structure)

    # Проверка первых 10 записей
    query_data = "SELECT * FROM stocks LIMIT 10;"
    data_sample = pd.read_sql_query(query_data, conn)
    print("\nПервые 10 записей:")
    print(data_sample)

    # Статистика по числовым колонкам (цены и объемы)
    query_stats = """
    SELECT 
        COUNT(*) as total_days,
        AVG(open) as avg_open,
        AVG(close) as avg_close,
        MIN(low) as min_low,
        MAX(high) as max_high,
        AVG(volume) as avg_volume,
        SUM(volume) as total_volume
    FROM stocks;
    """
    stats = pd.read_sql_query(query_stats, conn)
    print("\nБазовая статистика по акциям:")
    print(stats)

    # Временной диапазон данных (минимальная и максимальная дата)
    query_dates = """
    SELECT 
        MIN(date) as start_date, 
        MAX(date) as end_date 
    FROM stocks;
    """
    dates = pd.read_sql_query(query_dates, conn)
    print("\nПериод данных в БД:")
    print(dates)

    conn.close()
