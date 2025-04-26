import pandas as pd

# Замените 'your_file.xlsx' на путь к вашему Excel-файлу
file_path = '../../data/Агрегированные_данные_проценты.xlsx'

# Прочитайте Excel-файл (передайте имя листа, если необходимо)
df = pd.read_excel(file_path)

# Выведите тип данных каждого столбца
print(df.dtypes)

# Проверка дубликатов
print("\n=== Дубликаты ===")
print(f"Количество дубликатов: {df.duplicated().sum()}")
