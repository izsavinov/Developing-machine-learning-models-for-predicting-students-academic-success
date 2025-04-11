import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

def get_aggreagate_data_from_logs(file_path):
    df = pd.read_csv(file_path)

    # Исключение студентов "-" и "smartlms service"
    df = df[~df['Полное имя пользователя'].isin(["-", "smartlms service"])]

    # Преобразование столбца 'Дата' в формат datetime
    df['Время'] = pd.to_datetime(df['Время'], format='%d/%m/%y, %H:%M:%S')

    # Определение названия курса из самой ранней строки
    course_name = df.sort_values('Время')['Контекст события'].iloc[0]

    # Извлечение года из названия курса
    year_range = course_name.split("(")[1].split(" ")[0]  # Получаем "2023/2024"
    start_year, end_year = map(int, year_range.split("/"))  # Разделяем на 2023 и 2024

    # Определение максимального номера модуля
    max_module = int(course_name.split("модули:")[1].split(")")[0].split(",")[-1].strip())

    # Словарь с четвертями
    quarters_end = {
        "Q1": datetime(1, 10, 31),
        "Q2": datetime(1, 12, 30),
        "Q3": datetime(1, 3, 31),
        "Q4": datetime(1, 6, 1),
    }

    # Определение общего количества студентов
    total_students = df['Полное имя пользователя'].nunique()

    # Подсчет времени первого просмотра курса для каждого студента
    first_view_times = \
        df[df['Название события'].str.contains('Курс просмотрен', na=False)].groupby('Полное имя пользователя')['Время'].min()

    # Сортировка времён первого просмотра по возрастанию
    sorted_first_view_times = first_view_times.sort_values()

    # Поиск времени, когда более 5% студентов просмотрели курс
    cumulative_percentage = (sorted_first_view_times.rank(method='min') / total_students) * 100
    threshold_time = sorted_first_view_times[cumulative_percentage > 5].min()

    print(f"Более 5% студентов просмотрели курс к {threshold_time}. Это время принимается за start_time.")
    start_time = threshold_time.replace(year=start_year).normalize()

    # Определение конца курса
    end_time = quarters_end[f"Q{max_module}"].replace(year=end_year)

    # Вычисление временных меток для процентных интервалов
    percentages = [0.10, 0.25, 0.33, 0.50, 0.66, 0.8, 1]
    time_intervals = {f"{int(p * 100)}%": start_time + (end_time - start_time) * p for p in percentages}

    # Фильтрация событий
    df['Вход в курс'] = df['Название события'].str.contains('Курс просмотрен', na=False)
    df['Просмотр модуля'] = df['Название события'].str.contains('Модуль курса просмотрен', na=False)
    df['Просмотр ошибок'] = df['Название события'].str.contains('Попытка теста просмотрена', na=False)
    df['Просмотр своей оценки'] = df['Название события'].str.contains('Отзыв просмотрен', na=False)
    # df['Скачивание файла'] = df['Название события'].str.contains('Скачивание файла', na=False)
    df['Выполненные задания'] = df['Название события'].str.contains('Работа представлена', na=False)

    # Группировка данных по студенту
    grouped = df.groupby('Полное имя пользователя')

    # Создание пустого DataFrame для результатов
    result = pd.DataFrame(index=grouped.groups.keys())

    # Вычисление метрик для каждого процентного интервала
    for percentage, time_limit in time_intervals.items():
        filtered_df = df[df['Время'] <= time_limit]
        grouped_filtered = filtered_df.groupby('Полное имя пользователя')

        # Добавление метрик для текущего интервала
        result[f'Число входов в курс {percentage}'] = grouped_filtered['Вход в курс'].sum().reindex(result.index, fill_value=0)
        result[f'Число просмотров модулей {percentage}'] = grouped_filtered['Просмотр модуля'].sum().reindex(result.index, fill_value=0)
        result[f'Число просмотров своих ошибок {percentage}'] = grouped_filtered['Просмотр ошибок'].sum().reindex(result.index, fill_value=0)
        result[f'Количество выполненных заданий {percentage}'] = grouped_filtered['Выполненные задания'].sum().reindex(result.index, fill_value=0)
        # result[f'Число скачанных файлов {percentage}'] = grouped_filtered['Скачивание файла'].sum().reindex(result.index, fill_value=0)

        # Среднее время между заходами для текущего интервала
        result[f'Среднее время между заходами {percentage}'] = grouped_filtered.apply(
            lambda x: x.sort_values('Время')['Время'].diff().mean(),
            include_groups=False
        ).apply(lambda x: x.total_seconds() / 60 if pd.notnull(x) else 0).reindex(result.index, fill_value=0)

    return result

from bs4 import BeautifulSoup
import pandas as pd

def parsing_html_file(file_path):
    # Чтение HTML-файла
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Парсинг HTML с помощью BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Список для хранения данных
    data = []

    # Словарь для подсчета количества элементов контроля в каждой категории
    category_item_counts = {}

    # Первый проход: подсчет количества элементов контроля для каждой категории
    rows = soup.find_all('tr')
    current_category = None
    valid_categories = set()  # Множество для хранения допустимых категорий

    for row in rows:
        # Если строка является категорией
        if 'category' in row.get('class', []):
            # Извлечение названия категории
            category_name = row.find('td', class_='column-name')
            if category_name:
                current_category = category_name.get_text(strip=True)

            # Проверка наличия блока <label> с корректной структурой
            label_element = row.find('label', class_='m-0')
            if label_element and label_element.get_text(strip=True) != "Все":
                for_attr = label_element.get('for', '')
                if len(for_attr.split()) > 1:  # Проверяем, что атрибут for содержит несколько идентификаторов
                    valid_categories.add(current_category)  # Добавляем категорию в список допустимых
                    category_item_counts[current_category] = 0  # Инициализация счетчика элементов контроля

        # Если строка является элементом контроля
        elif 'item' in row.get('class', []):
            # Убедимся, что текущая категория определена и допустима
            if current_category and current_category in valid_categories:
                # Извлечение названия элемента контроля
                item_name_element = row.find(['a', 'span'], class_='gradeitemheader')  # Ищем как <a>, так и <span>
                if item_name_element:
                    # Извлекаем только текстовое содержимое, игнорируя вложенные теги
                    item_name = ''.join(item_name_element.find_all(text=True, recursive=False)).strip()
                else:
                    item_name = "Unnamed Item"

                # Если это "Итого в категории" или последующие элементы, не учитываем их
                if "Итого в категории" in item_name:
                    continue

                # Удаляем подстроку " (Значение)" из названия элемента
                item_name = item_name.replace(" (Значение)", "")

                # Проверяем наличие атрибута alt в теге <img>
                img_element = item_name_element.find('img')
                if img_element and 'alt' in img_element.attrs:
                    item_name += ":"

                # Увеличиваем счетчик элементов контроля для текущей категории
                category_item_counts[current_category] += 1

    # Второй проход: расчет весов
    current_category = None
    current_category_weight = None
    ignore_items = False  # Флаг для игнорирования элементов после "Итого в категории"

    for row in rows:
        # Если строка является категорией
        if 'category' in row.get('class', []):
            # Извлечение названия категории
            category_name = row.find('td', class_='column-name')
            if category_name:
                current_category = category_name.get_text(strip=True)
                ignore_items = False  # Сброс флага при переходе к новой категории

            # Извлечение веса категории
            weight_input = row.find('input', {'name': lambda x: x and x.startswith('weight_')})
            current_category_weight = (
                float(weight_input['value'].replace(',', '.'))
                if weight_input and 'value' in weight_input.attrs
                else None
            )

        # Если строка является элементом контроля
        elif 'item' in row.get('class', []):
            # Убедимся, что текущая категория определена и допустима
            if not current_category or current_category not in valid_categories:
                continue

            # Если флаг ignore_items активен, пропускаем все элементы контроля
            if ignore_items:
                continue

            # Извлечение названия элемента контроля
            item_name_element = row.find(['a', 'span'], class_='gradeitemheader')  # Ищем как <a>, так и <span>
            if item_name_element:
                # Извлекаем только текстовое содержимое, игнорируя вложенные теги
                item_name = ''.join(item_name_element.find_all(text=True, recursive=False)).strip()
            else:
                item_name = "Unnamed Item"

            # Если это "Итого в категории" или последующие элементы, не учитываем их
            if "Итого в категории" in item_name:
                ignore_items = True  # Активируем флаг, чтобы игнорировать последующие элементы
                continue  # Пропускаем этот элемент

            # Удаляем подстроку " (Значение)" из названия элемента
            # item_name = item_name.replace(" (Значение)", "")

            # Проверяем наличие атрибута alt в теге <img>
            img_element = item_name_element.find('img')
            if img_element and 'alt' in img_element.attrs:
                alt_text = img_element['alt'].strip()
                item_name = f"{alt_text}:{item_name}"

            item_name = f"{item_name}(Значение)"

            # Извлечение веса элемента контроля (если он указан)
            item_weight_input = row.find('input', {'id': lambda x: x and x.startswith('weight_')})
            item_weight = (
                float(item_weight_input['value'].replace(',', '.'))
                if item_weight_input and 'value' in item_weight_input.attrs
                else None
            )

            # Извлечение максимальной оценки
            max_score_element = row.find('td', class_='column-range')
            max_score = (
                float(max_score_element.get_text(strip=True).replace(',', '.'))
                if max_score_element and max_score_element.get_text(strip=True).replace(',', '').isdigit()
                else None
            )

            # Расчет итогового веса
            if item_weight is not None:
                final_weight = item_weight * current_category_weight  # Final Weight = Item Weight * Category Weight
            else:
                total_items = category_item_counts[current_category]
                final_weight = current_category_weight / total_items if current_category_weight else None

            # Добавляем данные в список
            data.append({
                "Category": current_category,
                "Item": item_name,
                "Category Weight": current_category_weight,
                "Item Weight": item_weight,
                "Final Weight": final_weight,
                "Max Score": max_score  # Добавляем максимальную оценку
            })

    # Создание DataFrame
    df = pd.DataFrame(data)

    return df

df_aggregate_data = get_aggreagate_data_from_logs('data/2_Логи.csv')

df_weights = parsing_html_file('data/Оценки_ Настройки0.html')

# Чтение данных из файла с оценками
file_path_grades = 'data/2_Оценки.xlsx'
df_grades = pd.read_excel(file_path_grades)

# Инициализация итоговой оценки
df_aggregate_data['Оценка за курс'] = 0.0

# Преобразуем все оценки в числа
for col in df_grades.columns:
    if col not in ['Фамилия', 'Имя', 'Адрес электронной почты', 'Индивидуальный номер', 'Данные о пользователе', 'User information', 'ID образовательной программы']:
        df_grades[col] = pd.to_numeric(df_grades[col], errors='coerce').fillna(0)

# Создаем словарь весов и максимальных оценок
weights_dict = df_weights.set_index('Item').to_dict(orient='index')

# Преобразуем индекс в столбец и приводим к строковому типу
df_aggregate_data = df_aggregate_data.reset_index()
df_aggregate_data['Полное имя пользователя'] = df_aggregate_data['index'].astype(str)
df_aggregate_data.drop('index', axis=1, inplace=True)

# Приводим к строковому типу в df_grades
df_grades['Полное имя пользователя'] = df_grades['Фамилия'].astype(str) + ' ' + df_grades['Имя'].astype(str)

# print(df_weights.head())
# print(df_aggregate_data.head())
# print(df_grades.head())

# Объединяем данные
df_aggregate_data = df_aggregate_data.merge(
    df_grades[['Полное имя пользователя']],
    on='Полное имя пользователя',
    how='inner'
)

print(len(df_aggregate_data))

# Сбрасываем индекс для правильной работы с данными
df_aggregate_data.reset_index(drop=True, inplace=True)

# Инициализация итоговой оценки
df_aggregate_data['Оценка за курс'] = 0.0

# Расчет итоговой оценки с объединением по ФИО
for element in df_weights['Item']:
    matching_columns = [col for col in df_grades.columns if element.strip().replace(" ", "") == col.strip().replace(" ", "")]

    if not matching_columns:
        print(f"Элемент '{element}' не найден")
        continue

    for col in matching_columns:
        if element not in weights_dict:
            print(f"Нет весов для '{element}'")
            continue

        max_score = weights_dict[element].get('Max Score', 0)
        weight = weights_dict[element].get('Final Weight', 0)

        # Создаем временный DataFrame для расчета
        temp_df = df_grades[['Полное имя пользователя', col]].copy()
        temp_df['contribution'] = (temp_df[col].astype(float) * 10 / max_score) * weight

        # Объединяем с основными данными
        df_aggregate_data = df_aggregate_data.merge(
            temp_df[['Полное имя пользователя', 'contribution']],
            on='Полное имя пользователя',
            how='left'
        )

        # Суммируем вклад
        df_aggregate_data['Оценка за курс'] += temp_df['contribution']
        df_aggregate_data.drop('contribution', axis=1, inplace=True)
        # print(df_aggregate_data['Оценка за курс'][0])


# Объединение таблиц внутренним соединением
# result = df_aggregate_data.merge(df_grades, left_index=True, right_on='Полное имя пользователя', how='inner')

# Перемещение ФИО студента на первое место
# result = df_aggregate_data.set_index('Полное имя пользователя').reset_index()

# Заполнение пропусков нулями
result = df_aggregate_data.fillna(0)
# print(df_aggregate_data['Оценка за курс'])

# Сохранение результата в новый Excel-файл
output_file = 'data/Агрегированные_данные_проценты.xlsx'
result.to_excel(output_file, index=False)

print(f"Агрегированные данные успешно сохранены в файл: {output_file}")