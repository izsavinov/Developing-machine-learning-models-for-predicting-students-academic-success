import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

def get_aggreagate_data_from_logs(file_path):
    df = pd.read_excel(file_path)

    # df = pd.read_excel(file_path)

    df['Время'] = df['Время'].astype(str) + ',' + df['ВремяДата'].astype(str)

    # Исключение студентов "-" и "smartlms service"
    df = df[~df['Полное имя пользователя'].isin(["-", "smartlms service"])]

    # Преобразование столбца 'Дата' в формат datetime
    df['Время'] = pd.to_datetime(df['Время'], format='%Y-%m-%d, %H:%M:%S')

    # Определение названия курса из самой ранней строки
    course_name = df.sort_values('Время')['Контекст события'].iloc[0]

    # Извлечение года из названия курса
    year_range = course_name.split("(")[1].split(" ")[0]  # Получаем "2023/2024"
    start_year, end_year = map(int, year_range.split("/"))  # Разделяем на 2023 и 2024

    # Определение максимального номера модуля
    modules = course_name.split("модули:")[1].split(")")[0].strip().split(",")
    modules = [int(m.strip()) for m in modules]
    min_module = min(modules)
    max_module = max(modules)

    # Словарь с четвертями
    quarters_end = {
        "Q1": datetime(1, 10, 31),
        "Q2": datetime(1, 12, 30),
        "Q3": datetime(1, 3, 31),
        "Q4": datetime(1, 6, 30),
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
    if min_module <= 2:
        # Если максимум 1 или 2 модуль, то конец курса в том же году
        start_time = start_time.replace(year=start_year)
    else:
        # Если 3 или 4 модуль, то конец курса в следующем году
        start_time = start_time.replace(year=end_year)

    # Определение конца курса
    if max_module <= 2:
        # Если максимум 1 или 2 модуль, то конец курса в том же году
        end_time = quarters_end[f"Q{max_module}"].replace(year=start_year)
    else:
        # Если 3 или 4 модуль, то конец курса в следующем году
        end_time = quarters_end[f"Q{max_module}"].replace(year=end_year)

    # Вычисление временных меток для процентных интервалов
    percentages = [0.0, 0.25, 0.33, 0.50, 0.66, 1.0]  # Добавляем 0.0 как начальную точку
    intervals = []
    for i in range(1, len(percentages)):
        start_p = percentages[i - 1]
        end_p = percentages[i]
        intervals.append((start_p, end_p))

    # Вычисление временных меток для интервалов
    time_intervals = []
    for start_p, end_p in intervals:
        start_interval = start_time + (end_time - start_time) * start_p
        end_interval = start_time + (end_time - start_time) * end_p
        label = f"{int(end_p * 100)}%"
        time_intervals.append((label, start_interval, end_interval))

    # Фильтрация событий
    df['Вход в курс'] = df['Название события'].str.contains('Курс просмотрен', na=False)
    df['Просмотр модуля'] = df['Название события'].str.contains('Модуль курса просмотрен', na=False)
    df['Просмотр ошибок'] = df['Название события'].str.contains('Попытка теста просмотрена', na=False)
    df['Просмотр своей оценки'] = df['Название события'].str.contains('Отзыв просмотрен', na=False)
    df['Выполненные задания'] = df['Название события'].isin(['Работа представлена.', 'Попытка теста завершена и отправлена на оценку'])
    df['Оцененные задания'] = df['Название события'].str.contains('Пользователю поставлена оценка', na=False)

    # Группировка данных по студенту
    grouped = df.groupby('Полное имя пользователя')

    # Создание пустого DataFrame для результатов
    result = pd.DataFrame(index=grouped.groups.keys())

    # Вычисление метрик для каждого интервала
    for label, start_interval, end_interval in time_intervals:
        # Фильтрация данных для обычных метрик (текущий интервал)
        filtered_df_current = df[(df['Время'] >= start_interval) & (df['Время'] < end_interval)]

        # Фильтрация данных для выполненных/оцененных заданий (кумулятивно: 0% → текущий интервал)
        filtered_df_cumulative = df[df['Время'] < end_interval]

        grouped_filtered_current = filtered_df_current.groupby('Полное имя пользователя')

        # Добавление метрик с указанием интервала
        # 1. Метрики ТОЛЬКО для текущего интервала
        result[f'Число входов в курс {label}'] = grouped_filtered_current['Вход в курс'].sum().reindex(result.index,
                                                                                                       fill_value=0)
        result[f'Число просмотров модулей {label}'] = grouped_filtered_current['Просмотр модуля'].sum().reindex(
            result.index, fill_value=0)
        result[f'Число просмотров своих ошибок {label}'] = grouped_filtered_current['Просмотр ошибок'].sum().reindex(
            result.index, fill_value=0)
        result[f'Число просмотров полученных оценок {label}'] = grouped_filtered_current['Просмотр своей оценки'].sum().reindex(
            result.index, fill_value=0)
        result[f'Количество выполненных заданий {label}'] = grouped_filtered_current[
            'Выполненные задания'].sum().reindex(result.index, fill_value=0)
        result[f'Среднее время между заходами {label}'] = (
            grouped_filtered_current['Время']
            .apply(lambda x: x.sort_values().diff().mean())
            .apply(lambda x: x.total_seconds() / 60 if pd.notnull(x) else 0)
            .reindex(result.index, fill_value=0)
        )

        # 2. Кумулятивные метрики (0% → текущий интервал)
        # Сбор выполненных заданий (нарастающий итог)
        # Для логов 1_
        filtered_df_cumulative['ФИО'] = filtered_df_cumulative.apply(
            lambda row: row['Полное имя пользователя'] if row[
                                                              'Название события'] != 'Выполнение элемента курса обновлено' else
            row['Затронутый пользователь'],
            axis=1
        )

        completed_assignments = (
            filtered_df_cumulative[filtered_df_cumulative['Название события'].isin(
                ['Работа представлена.', 'Попытка теста завершена и отправлена на оценку',
                 'Выполнение элемента курса обновлено'])]
            .groupby('ФИО')['Контекст события']
            .apply(lambda x: '(,)'.join(x.unique()))
            .str.replace(" ", "")
            .reindex(result.index, fill_value='')
        )

        #Для логов 4_
        # filtered_df_cumulative['ФИО'] = filtered_df_cumulative.apply(
        #     lambda row: row['Полное имя пользователя'] if row[
        #                                                       'Название события'] != 'Представленный ответ был оценен.' else
        #     row['Затронутый пользователь'],
        #     axis=1
        # )
        #
        # completed_assignments = (
        #     filtered_df_cumulative[filtered_df_cumulative['Название события'].isin(
        #         ['Работа представлена.', 'Попытка теста завершена и отправлена на оценку',
        #          'Представленный ответ был оценен.'])]
        #     .groupby('ФИО')['Контекст события']
        #     .apply(lambda x: '(,)'.join(x.unique()))
        #     .str.replace(" ", "")
        #     .reindex(result.index, fill_value='')
        # )

        result[f'Выполненные задания {label}'] = completed_assignments

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
            if label_element == None or label_element.get_text(strip=True) != "Все":
                # for_attr = label_element.get('for', '')
                # if len(for_attr.split()) > 1:  # Проверяем, что атрибут for содержит несколько идентификаторов
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

            if current_category_weight == None:
                current_category_weight = 1

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

df_aggregate_data = get_aggreagate_data_from_logs('../../data/raw/7_Логи.xlsx')

# df_weights = pd.read_excel('../../data/5_weights.xlsx')

# Чтение данных из файла с оценками
file_path_grades = '../../data/raw/7_Оценки.xlsx'
df_grades = pd.read_excel(file_path_grades)

# print(df_grades.columns)

num_of_students = len(df_grades)

# Преобразуем все оценки в числа
for col in df_grades.columns:
    if col not in ['ФИО', 'Адрес электронной почты', 'Индивидуальный номер', 'Данные о пользователе', 'User information', 'ID образовательной программы']:
        df_grades[col] = pd.to_numeric(df_grades[col], errors='coerce').fillna(0)

# Создаем словарь весов и максимальных оценок
# weights_dict = df_weights.set_index('Item').to_dict(orient='index')
#
# weights_dict = {
#     key.replace("(Значение)", "").replace(" ", ""): value
#     for key, value in weights_dict.items()
# }

# print(weights_dict)

# Преобразуем индекс в столбец и приводим к строковому типу
df_aggregate_data = df_aggregate_data.reset_index()
df_aggregate_data['Полное имя пользователя'] = df_aggregate_data['index'].astype(str)
df_aggregate_data.drop('index', axis=1, inplace=True)

# Приводим к строковому типу в df_grades
df_grades['Полное имя пользователя'] = df_grades['ФИО'].astype(str) # + ' ' + df_grades['Имя'].astype(str)

# Для добавления оценок из df_grades в df_aggregate_data
# Нужно объединить таблицы так, чтобы все необходимые поля из df_grades появились в df_aggregate_data

# Шаг 1. Объединяем по полю 'Полное имя пользователя' с нужными колонками
def normalize_name(name):
    # Заменяет 'ё' на 'е' в имени
    return str(name).replace('ё', 'е').replace('Ё', 'Е')

# Добавим нормализованное имя в оба датафрейма
df_aggregate_data['Нормализованное ФИО'] = df_aggregate_data['Полное имя пользователя'].apply(normalize_name)
df_grades['Нормализованное ФИО'] = df_grades['Полное имя пользователя'].apply(normalize_name)

# Соединяем по нормализованному полю
df_aggregate_data = df_aggregate_data.merge(
    df_grades[['Нормализованное ФИО',
               'Лекции1', 'Семинары1', 'Лабы1', 'ДЗ1',
               'Лекции2', 'Семинары2', 'Лабы2', 'ДЗ2']],
    left_on='Нормализованное ФИО',
    right_on='Нормализованное ФИО',
    how='inner'
)

# По желанию удаляем вспомогательное поле
df_aggregate_data = df_aggregate_data.drop(columns=['Нормализованное ФИО'])

# Шаг 2. Заполняем оценки по этапам формулами
df_aggregate_data['Оценка за интервал 25%'] = (
    0.5 * 0.05 * df_aggregate_data['Лекции1'] +
    0.5 * 0.05 * df_aggregate_data['Семинары1'] +
    0.5 * 0.2  * df_aggregate_data['Лабы1']
)

df_aggregate_data['Оценка за интервал 33%'] = (
    0.66 * 0.05 * df_aggregate_data['Лекции1'] +
    0.66 * 0.05 * df_aggregate_data['Семинары1'] +
    0.66 * 0.2  * df_aggregate_data['Лабы1']
)

df_aggregate_data['Оценка за интервал 50%'] = (
    0.05 * df_aggregate_data['Лекции1'] +
    0.05 * df_aggregate_data['Семинары1'] +
    0.2  * df_aggregate_data['Лабы1'] +
    0.2  * df_aggregate_data['ДЗ1']
)

df_aggregate_data['Оценка за интервал 66%'] = (
    df_aggregate_data['Оценка за интервал 50%'] +
    0.32 * 0.05 * df_aggregate_data['Лекции2'] +
    0.32 * 0.05 * df_aggregate_data['Семинары2'] +
    0.32 * 0.2  * df_aggregate_data['Лабы2']
)

df_aggregate_data['Оценка за интервал 100%'] = (
    df_aggregate_data['Оценка за интервал 50%'] +
    0.05 * df_aggregate_data['Лекции2'] +
    0.05 * df_aggregate_data['Семинары2'] +
    0.2  * df_aggregate_data['Лабы2'] +
    0.2  * df_aggregate_data['ДЗ2']
)

# После расчётов удаляем ненужные поля с оценками

fields_to_drop = [
    'Лекции1', 'Семинары1', 'Лабы1', 'ДЗ1',
    'Лекции2', 'Семинары2', 'Лабы2', 'ДЗ2'
]

df_aggregate_data = df_aggregate_data.drop(columns=fields_to_drop)


# Сбрасываем индекс для правильной работы с данными
df_aggregate_data.reset_index(drop=True, inplace=True)

# Инициализация итоговой оценки
# df_aggregate_data['Оценка за курс'] = 0.0

# Расчет итоговой оценки с объединением по ФИО
# Для каждой колонки с выполненными заданиями и оцененными заданиями
# for col in df_aggregate_data.columns:
#     if col.startswith("Выполненные задания") or col.startswith("Оцененные задания"):
#         # Инициализация столбца для оценки за интервал
#         interval_percentage = col.split()[-1]  # Например, "10%"
#         df_aggregate_data[f'Оценка за интервал {interval_percentage}'] = (
#             df_aggregate_data.get(f'Оценка за интервал {interval_percentage}', 0)
#         )
#
#         # Для каждого студента в df_aggregate_data
#         for index, row in df_aggregate_data.iterrows():
#             assignments_str = row[col]  # Список выполненных или оцененных заданий через запятую
#             if not isinstance(assignments_str, str):
#                 continue  # Пропускаем, если нет данных
#
#             # Разделяем названия заданий по запятой
#             assignments_list = [assignment.strip() for assignment in assignments_str.split("(,)")]
#
#             if len(assignments_list) == 0:
#                 continue
#
#             # Рассчитываем вклад для каждого задания
#             for element in assignments_list:
#                 if element == "":
#                     continue
#
#                 # Если это колонка "Оцененные задания", используем ID из df_weights
#                 # if col.startswith("Оцененные задания"):
#                 #     # Ищем элемент в df_weights по Item ID
#                 #     matching_item = None
#                 #     for item_name, item_data in weights_dict.items():
#                 #         if str(item_data.get('Item ID')) == element:
#                 #             matching_item = item_name
#                 #             break
#                 #
#                 #     if not matching_item:
#                 #         # print(f"Элемент с ID '{element}' не найден в weights_dict")
#                 #         continue
#                 #
#                 #     # Получаем название элемента, максимальную оценку и вес из weights_dict
#                 #     element_name = matching_item
#                 #     max_score = weights_dict[element_name].get('Max Score', 0)
#                 #     weight = weights_dict[element_name].get('Final Weight', 0)
#                 #
#                 # else:
#                 #     Если это колонка "Выполненные задания", используем название элемента напрямую
#                 element_name = element
#                 if element_name not in weights_dict:
#                     print(f"Нет весов для '{element_name}'")
#                     continue
#
#                 max_score = weights_dict[element_name].get('Max Score', 0)
#                 weight = weights_dict[element_name].get('Final Weight', 0)
#                     # continue
#
#                 # Проверяем, есть ли соответствующая колонка в df_grades
#                 matching_columns = [
#                     col_grade for col_grade in df_grades.columns
#                     if element_name.replace(" ", "") == col_grade.replace(" ", "")
#                 ]
#
#                 if not matching_columns:
#                     print(f"Элемент '{element_name}' не найден в df_grades")
#                     continue
#
#                 # Создаем временный DataFrame для расчета
#                 temp_df = df_grades[['Полное имя пользователя', matching_columns[0]]].copy()
#                 temp_df['contribution'] = (temp_df[matching_columns[0]].astype(float) * 10 / max_score) * weight
#
#                 # Фильтруем только текущего студента
#                 student_contribution = temp_df[temp_df['Полное имя пользователя'] == row['Полное имя пользователя']]['contribution'].sum()
#
#                 # Добавляем вклад к оценке за интервал
#                 df_aggregate_data.at[index, f'Оценка за интервал {interval_percentage}'] += student_contribution

# Заполнение пропусков нулями
result = df_aggregate_data.fillna(0)

# Создаем новый список с нужным порядком колонок
columns = ["Полное имя пользователя"] + [col for col in result if col != "Полное имя пользователя"]
# Переупорядочиваем DataFrame
result = result[columns]

# Нормализация числовых метрик по максимальной активности группы для каждого периода
numeric_columns = [
    col for col in result.columns
    if any(metric in col for metric in [
        'Число входов в курс',
        'Число просмотров модулей',
        'Число просмотров своих ошибок',
        'Число просмотров полученных оценок',
        'Количество выполненных заданий',
        'Среднее время между заходами'
    ])
]

# Список уникальных периодов (например, 25%, 50%, 75%)
periods = sorted(set([col.split()[-1] for col in numeric_columns]))

# Нормализация для каждого периода
for period in periods:
    # Выбор столбцов, относящихся к текущему периоду
    period_columns = [col for col in numeric_columns if col.endswith(period)]

    # Создание подтаблицы для текущего периода
    period_data = result[period_columns]

    # Вычисление максимальных значений для каждой колонки
    max_values = period_data.max(axis=0)  # Максимальные значения для каждого столбца

    # Проверка максимумов
    for col, max_val in max_values.items():
        if max_val == 0:
            print(f"Внимание: В колонке '{col}' все значения равны 0. Колонка остается без изменений.")
            result[col] = 0  # Все значения в колонке остаются равными 0
        elif max_val < 1:
            print(f"Внимание: В колонке '{col}' максимальное значение меньше 1 ({max_val}). Проверьте данные.")
        else:
            # Нормализация значений внутри колонки
            result[col] = period_data[col] / max_val

    # Проверка: Убедимся, что в каждой колонке есть хотя бы одно значение 1
    for col in period_columns:
        if result[col].max() != 1 and max_values[col] != 0:  # Исключаем колонки с нулевым максимумом
            print(f"Внимание: В колонке '{col}' нет значения 1 после нормализации.")

# Сохранение результата в новый Excel-файл
output_file = '../../data/7_Агрегированные_данные_проценты.xlsx'
result.to_excel(output_file, index=False)

print(f"Агрегированные данные успешно сохранены в файл: {output_file}")