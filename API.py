from flask import Flask, request, jsonify, make_response
import pandas as pd
import os
import json
from collections import OrderedDict
import base64
import joblib
from datetime import datetime

app = Flask(__name__)

# Загрузка метаданных о курсах
COURSES_METADATA_PATH = "../data/courses_metadata.xlsx"

# Загрузка паролей студентов
PASSWORDS_PATH = "../data/passwords.xlsx"

def load_courses_metadata():
    return pd.read_excel(COURSES_METADATA_PATH)

def load_passwords():
    """Загружает данные о пользователях (ФИО и пароли)."""
    return pd.read_excel(PASSWORDS_PATH)

def get_predictive_assessment(df, current_period, fio):
    model_data = joblib.load(f'best_knn_model_period_{current_period}%.joblib')
    loaded_model = model_data['model']
    required_columns = model_data['features']
    scaler = model_data['scaler']
    pca = model_data['pca']

    # Находим строку по ФИО
    user_row = df[df["Полное имя пользователя"] == fio]

    # Проверяем, что все нужные колонки есть
    if not all(col in df.columns for col in required_columns):
        print("Ошибка: В данных отсутствуют некоторые колонки.")
        return None, None

    # Проверяем, найден ли пользователь
    if user_row.empty:
        print(f"Пользователь {fio} не найден в данных.")
        user_pred = None
    else:
        X_user = user_row[required_columns]
        # 1. Масштабирование
        if scaler is not None:
            X_user = scaler.transform(X_user)  # Используем сохраненный scaler  # Получим данные с уменьшенной размерностью

        if pca is not None:
            X_user = pca.transform(X_user)  # Получим данные с уменьшенной размерностью

        if model_data['pca'] is not None:
            X_user = model_data['pca'].transform(X_user)

        user_pred = loaded_model.predict(X_user)[0]

    X_all = df[required_columns]
    # 1. Масштабирование
    if scaler is not None:
        X_all = scaler.transform(X_all)  # Используем сохраненный scaler  # Получим данные с уменьшенной размерностью

    if pca is not None:
        X_all = pca.transform(X_all)  # Получим данные с уменьшенной размерностью

    all_preds = loaded_model.predict(X_all)
    mean_pred = all_preds.mean()

    return user_pred, mean_pred

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/login', methods=['POST'])
def login():
    """
    Авторизует пользователя и записывает ФИО и пароль в куки.
    """
    try:
        # Получение данных из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return jsonify({"error": "Необходимо указать учетные данные в Basic Auth"}), 401

        # Декодирование Base64 строки
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        fio, password = decoded_credentials.split(":")

        # Загрузка данных о пользователях (ФИО и пароли)
        passwords_df = load_passwords()
        user_record = passwords_df[passwords_df["ФИО"] == fio]

        # Проверка авторизации
        if user_record.empty or str(user_record.iloc[0]["Пароль"]) != str(password):
            return jsonify({"error": "Неверные ФИО или пароль"}), 401

        # Создание ответа с куками
        response = make_response(jsonify({"message": "Авторизация успешна"}), 200)
        response.set_cookie("fio", fio, httponly=True)  # Установка куки для ФИО
        response.set_cookie("password", base64.b64encode(password.encode()).decode(), httponly=True)  # Установка зашифрованного пароля
        return response

    except Exception as e:
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500

@app.route('/get_available_courses', methods=['POST'])
def get_available_courses():
    """
    Возвращает список всех актуальных курсов для текущего студента.
    Использует Basic Auth для авторизации.
    """
    try:
        # Получение данных из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return jsonify({"error": "Необходимо указать учетные данные в Basic Auth"}), 401

        # Декодирование Base64 строки
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        fio, password = decoded_credentials.split(":")

        # Загрузка данных о пользователях (ФИО и пароли)
        passwords_df = load_passwords()
        user_record = passwords_df[passwords_df["ФИО"] == fio]

        # Проверка авторизации
        if user_record.empty or str(user_record.iloc[0]["Пароль"]) != str(password):
            return jsonify({"error": "Неверные ФИО или пароль"}), 401

        # Загрузка метаданных о курсах
        metadata_df = load_courses_metadata()

        # Текущая дата
        current_date = datetime.now()

        # Поиск актуальных курсов для данного студента
        student_courses = metadata_df[
            (metadata_df["ФИО"] == fio) &
            (pd.to_datetime(metadata_df["ДатаНачала"]) <= current_date) &
            (pd.to_datetime(metadata_df["ДатаОкончания"]) >= current_date)
        ]

        # Преобразование в список курсов
        available_courses = list(student_courses['Курс'])

        # Возвращаем ответ в формате JSON
        return jsonify({
            "fio": fio,
            "available_courses": available_courses
        })

    except Exception as e:
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500

@app.route('/get_course_data', methods=['POST'])
def get_course_data():
    """
    Обрабатывает запрос от фронтенда и возвращает данные по выбранному курсу.
    """
    try:
        # Получение данных из запроса

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return jsonify({"error": "Необходимо указать учетные данные в Basic Auth"}), 401

        # Декодирование Base64 строки
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        fio, password = decoded_credentials.split(":")

        data = request.json
        course_name = data.get("course")

        if not fio or not course_name:
            return jsonify({"error": "Необходимо указать ФИО, пароль и курс"}), 400

        # Загрузка данных о пользователях (ФИО и пароли)
        passwords_df = load_passwords()
        user_record = passwords_df[passwords_df["ФИО"] == fio]

        # Проверка авторизации
        if user_record.empty or str(user_record.iloc[0]["Пароль"]) != str(password):
            return jsonify({"error": "Неверные ФИО или пароль"}), 401

        # Загрузка метаданных о курсах
        metadata_df = load_courses_metadata()
        student_courses = metadata_df[metadata_df["ФИО"] == fio]

        # Поиск информации о курсе
        course_info = student_courses[student_courses["Курс"] == course_name]
        if course_info.empty:
            return jsonify({"error": "Курс не найден для данного студента"}), 404

        # Чтение данных из Excel-файла
        file_path = course_info.iloc[0]["ПутьКФайлу"]
        if not os.path.exists(file_path):
            return jsonify({"error": "Файл с данными не найден"}), 404

        df = pd.read_excel(file_path)

        # Фильтрация данных для данного студента
        student_data = df[df["Полное имя пользователя"] == fio].to_dict(orient="records")[0]

        percents = [25, 33, 50, 66]

        metric_labels = [
            "Число просмотров модулей",
            "Число входов в курс",
            "Число просмотров своих ошибок",
            "Число просмотров полученных оценок",
            "Количество выполненных заданий",
            "Среднее время между заходами"
        ]

        # Фильтрация периодов: исключаем те, где все значения предикторов равны нулю
        valid_periods = []
        for percent in percents:
            # Проверяем, есть ли хотя бы одно ненулевое значение среди всех метрик для данного периода
            has_non_zero = any(
                student_data.get(f"{label} {percent}%", 0) != 0
                for label in metric_labels
            )
            if has_non_zero:
                valid_periods.append(percent)

        current_period = max(valid_periods)

        user_pred, mean_pred = get_predictive_assessment(df, current_period, fio)

        response = OrderedDict()
        response["fio"] = fio
        response["course_name"] = course_name
        response["period"] = f'{current_period}%'
        response["current_assessment"] = student_data.get(f'Оценка за интервал {current_period}%')
        response["current_assessment_mean"] = df[f'Оценка за интервал {current_period}%'].mean()
        response["predictive_assessment"] = user_pred
        response["predictive_assessment_mean"] = mean_pred

        charts_data = OrderedDict()
        for label in metric_labels:
            student_points = [
                {"period": percent, "value": student_data.get(f"{label} {percent}%")}
                for percent in valid_periods
            ]
            group_average_points = [
                {"period": percent, "value": df[f"{label} {percent}%"].mean()}
                for percent in valid_periods
            ]
            chart_key = label.lower().replace(' ', '_').replace('ё', 'e')
            charts_data[chart_key] = {
                "student": student_points,
                "group_average": group_average_points
            }

        response["charts"] = charts_data

        # Возвращаем через Response, чтобы не сломать порядок
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )

    except Exception as e:
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)