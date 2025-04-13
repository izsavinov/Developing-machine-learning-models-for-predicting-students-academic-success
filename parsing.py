import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import re

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
    first_view_times = (
        df[df['Название события'].str.contains('Курс просмотрен', na=False)]
        .groupby('Полное имя пользователя')['Время']
        .min()
    )

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
    df['Выполненные задания'] = df['Название события'].isin(['Работа представлена', 'Попытка теста завершена и отправлена на оценку'])
    df['Оцененные задания'] = df['Название события'].str.contains('Пользователю поставлена оценка', na=False)

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

        # Сбор названий выполненных заданий
        completed_assignments = (
            filtered_df[filtered_df['Название события'].isin(
                ['Работа представлена', 'Попытка теста завершена и отправлена на оценку'])]
            .groupby('Затронутый пользователь')['Контекст события']
            .apply(lambda x: '(,)'.join(x.unique()))
            .str.replace(" ", "")  # удаляем пробелы
            .reindex(result.index, fill_value='')
        )
        result[f'Выполненные задания {percentage}'] = completed_assignments

        # Сбор айди оцененных заданий
        graded_assignments = (
            filtered_df[filtered_df['Название события'].str.contains('Пользователю поставлена оценка', na=False)]
            .copy()
        )
        graded_assignments['Grade Item ID'] = graded_assignments['Описание'].apply(
            lambda x: re.search(r"for the grade item with id '(\d+)'", x).group(1) if re.search(r"for the grade item with id '(\d+)'", x) else None
        )
        graded_assignments_ids = (
            graded_assignments.groupby('Затронутый пользователь')['Grade Item ID']
            .apply(lambda x: '(,)'.join(x.dropna().unique()))
            .reindex(result.index, fill_value='')
        )
        result[f'Оцененные задания {percentage}'] = graded_assignments_ids

        # Среднее время между заходами для текущего интервала
        result[f'Среднее время между заходами {percentage}'] = grouped_filtered.apply(
            lambda x: x.sort_values('Время')['Время'].diff().mean(),
            include_groups=False
        ).apply(lambda x: x.total_seconds() / 60 if pd.notnull(x) else 0).reindex(result.index, fill_value=0)

    return result

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

            # Извлечение ID элемента контроля
            item_id = row.get('data-itemid', None)

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
            item_name = item_name.replace(" (Значение)", "").replace(" ", "")

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
                "Item ID": item_id,  # Добавляем ID элемента контроля
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

# print(df_weights['Item'])

# Чтение данных из файла с оценками
file_path_grades = 'data/2_Оценки.xlsx'
df_grades = pd.read_excel(file_path_grades)

# Обновляем только те заголовки, которые содержат '(Значение)'
updated_columns = [
    col.replace(" (Значение)", "").replace(" ", "")
    if "(Значение)" in col else col
    for col in df_grades.columns
]
df_grades.columns = updated_columns

print(df_grades.columns)

num_of_students = len(df_grades)

# Преобразуем все оценки в числа
for col in df_grades.columns:
    if col not in ['Фамилия', 'Имя', 'Адрес электронной почты', 'Индивидуальный номер', 'Данные о пользователе', 'User information', 'ID образовательной программы']:
        df_grades[col] = pd.to_numeric(df_grades[col], errors='coerce').fillna(0)

# Создаем словарь весов и максимальных оценок
weights_dict = df_weights.set_index('Item').to_dict(orient='index')

weights_dict = {
    key.replace("(Значение)", "").replace(" ", ""): value
    for key, value in weights_dict.items()
}

print(weights_dict.keys())

# Преобразуем индекс в столбец и приводим к строковому типу
df_aggregate_data = df_aggregate_data.reset_index()
df_aggregate_data['Полное имя пользователя'] = df_aggregate_data['index'].astype(str)
df_aggregate_data.drop('index', axis=1, inplace=True)

# Приводим к строковому типу в df_grades
df_grades['Полное имя пользователя'] = df_grades['Фамилия'].astype(str) + ' ' + df_grades['Имя'].astype(str)

# Объединяем данные
df_aggregate_data = df_aggregate_data.merge(
    df_grades[['Полное имя пользователя']],
    on='Полное имя пользователя',
    how='inner'
)

# Сбрасываем индекс для правильной работы с данными
df_aggregate_data.reset_index(drop=True, inplace=True)

# Инициализация итоговой оценки
# df_aggregate_data['Оценка за курс'] = 0.0

# Расчет итоговой оценки с объединением по ФИО
# Для каждой колонки с выполненными заданиями и оцененными заданиями
for col in df_aggregate_data.columns:
    if col.startswith("Выполненные задания") or col.startswith("Оцененные задания"):
        # Инициализация столбца для оценки за интервал
        interval_percentage = col.split()[-1]  # Например, "10%"
        df_aggregate_data[f'Оценка за интервал {interval_percentage}'] = (
            df_aggregate_data.get(f'Оценка за интервал {interval_percentage}', 0)
        )

        # Для каждого студента в df_aggregate_data
        for index, row in df_aggregate_data.iterrows():
            assignments_str = row[col]  # Список выполненных или оцененных заданий через запятую
            if not isinstance(assignments_str, str):
                continue  # Пропускаем, если нет данных

            # Разделяем названия заданий по запятой
            assignments_list = [assignment.strip() for assignment in assignments_str.split("(,)")]

            if len(assignments_list) == 0:
                continue

            # Рассчитываем вклад для каждого задания
            for element in assignments_list:
                if element == "":
                    continue

                # Если это колонка "Оцененные задания", используем ID из df_weights
                if col.startswith("Оцененные задания"):
                    # Ищем элемент в df_weights по Item ID
                    matching_item = None
                    for item_name, item_data in weights_dict.items():
                        if str(item_data.get('Item ID')) == element:
                            matching_item = item_name
                            break

                    if not matching_item:
                        # print(f"Элемент с ID '{element}' не найден в weights_dict")
                        continue

                    # Получаем название элемента, максимальную оценку и вес из weights_dict
                    element_name = matching_item
                    max_score = weights_dict[element_name].get('Max Score', 0)
                    weight = weights_dict[element_name].get('Final Weight', 0)

                else:
                    # Если это колонка "Выполненные задания", используем название элемента напрямую
                    # element_name = element
                    # if element_name not in weights_dict:
                    #     # print(f"Нет весов для '{element_name}'")
                    #     continue
                    #
                    # max_score = weights_dict[element_name].get('Max Score', 0)
                    # weight = weights_dict[element_name].get('Final Weight', 0)
                    continue

                # Проверяем, есть ли соответствующая колонка в df_grades
                matching_columns = [
                    col_grade for col_grade in df_grades.columns
                    if element_name.replace(" ", "") == col_grade.replace(" ", "")
                ]

                if not matching_columns:
                    print(f"Элемент '{element_name}' не найден в df_grades")
                    continue

                # Создаем временный DataFrame для расчета
                temp_df = df_grades[['Полное имя пользователя', matching_columns[0]]].copy()
                temp_df['contribution'] = (temp_df[matching_columns[0]].astype(float) * 10 / max_score) * weight

                # Фильтруем только текущего студента
                student_contribution = temp_df[temp_df['Полное имя пользователя'] == row['Полное имя пользователя']]['contribution'].sum()

                # Добавляем вклад к оценке за интервал
                df_aggregate_data.at[index, f'Оценка за интервал {interval_percentage}'] += student_contribution

# Заполнение пропусков нулями
result = df_aggregate_data.fillna(0)

# Сохранение результата в новый Excel-файл
output_file = 'data/Агрегированные_данные_проценты.xlsx'
result.to_excel(output_file, index=False)

print(f"Агрегированные данные успешно сохранены в файл: {output_file}")