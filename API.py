from flask import Flask, request, jsonify, make_response
import pandas as pd
import os
import json
from collections import OrderedDict
import base64
import joblib
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import traceback # Добавим для вывода стека ошибок

app = Flask(__name__)

# Загрузка метаданных о курсах
COURSES_METADATA_PATH = "../data/courses_metadata.xlsx"

# Загрузка паролей студентов
PASSWORDS_PATH = "../data/passwords.xlsx"

BEST_MODEL_PATHS_PER_PERIOD = {
    "25%": "saved_tcn_models_cv/inference_tcn_period_25pct_pca_True.pth",
    "33%": "best_knn_model_period_33%.joblib",
    "50%": "saved_tcn_models_cv/inference_tcn_period_50pct_pca_True.pth",
    "66%": "best_knn_model_period_66%.joblib",
}

def load_courses_metadata():
    return pd.read_excel(COURSES_METADATA_PATH)

def load_passwords():
    """Загружает данные о пользователях (ФИО и пароли)."""
    return pd.read_excel(PASSWORDS_PATH)

class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation_fn, dropout_rate):
        super().__init__()
        def _get_act(name): # Простая локальная версия get_activation_fn
            if name == 'ReLU': return nn.ReLU
            if name == 'GELU': return nn.GELU
            if name == 'LeakyReLU': return nn.LeakyReLU
            raise ValueError(f"Unknown activation: {name}")
        Activation = _get_act(activation_fn)
        self.causal_padding = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.causal_padding, 0), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = Activation()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.pad(x); out = self.conv(out); out = self.norm(out)
        out = self.activation(out); out = self.dropout(out); return out

# --- Модель TCN с Каузальными Свертками ---
class SequentialTCNModel(nn.Module):
    def __init__(self, num_input_channels_model, output_size, num_channels, kernel_size, dropout_rate, activation_fn, dilation_base=2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = num_input_channels_model
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            out_channels = num_channels[i]
            layers.append( CausalConv1dBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, activation_fn=activation_fn, dropout_rate=dropout_rate) )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.output_channels = num_channels[-1]
        self.output_layer = nn.Linear(self.output_channels, output_size)

    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        out = self.output_layer(out)
        return out
# >>> КОНЕЦ КОПИРОВАНИЯ ОПРЕДЕЛЕНИЙ КЛАССОВ <<<

# Класс LSTM модели
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM( input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional )
        linear_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(linear_input_dim, output_size)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x): # Ожидает (N, L, C)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout_layer(last_out)
        out = self.fc(out)
        return out

def get_predictive_assessment(df, model_path, fio):
    """
    Загружает модель (sklearn, TCN или LSTM) и делает предсказание.
    Args:
        df (pd.DataFrame): DataFrame с текущими данными студентов.
        model_path (str): Путь к сохраненному файлу модели (.joblib или .pth).
        fio (str): Полное имя пользователя для индивидуального прогноза.
    Returns:
        tuple: (user_prediction, mean_prediction) или (None, None) при ошибке.
    """
    print(f"Загрузка модели и данных из: {model_path}")
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден: {model_path}"); return None, None

    # Определяем тип файла
    is_torch_model = model_path.lower().endswith('.pth')
    is_sklearn_model = model_path.lower().endswith('.joblib')
    if not is_torch_model and not is_sklearn_model:
        print(f"Ошибка: Неизвестный тип файла: {model_path}"); return None, None

    # Инициализация переменных
    loaded_model = None; required_columns = None; scaler = None; pca = None
    metadata = None; model_type = None; inference_device = None

    try:
        # --- Загрузка ---
        if is_sklearn_model:
            saved_data = joblib.load(model_path)
            loaded_model = saved_data['model']
            required_columns = saved_data['features']
            scaler = saved_data.get('scaler')
            pca = saved_data.get('pca')
            model_type = "sklearn"
            print("Загружена модель Scikit-learn.")

        elif is_torch_model:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Инференс PyTorch модели на: {inference_device}")
            # Загружаем ВСЕ данные из файла
            saved_data = torch.load(model_path, map_location=inference_device, weights_only=False)

            # Извлекаем компоненты (независимо от того, TCN или LSTM)
            model_state_dict = saved_data['model_state_dict']
            model_class_name = saved_data['model_class_name']
            model_init_params = saved_data['model_init_params']
            required_columns = saved_data['feature_columns_ordered']
            scaler = saved_data['scaler']
            pca = saved_data.get('pca') # Может быть None
            metadata = saved_data['metadata'] # Содержит важные параметры для reshape/PCA

            print(f"Обнаружен класс модели: {model_class_name}")

            # --- Воссоздание модели PyTorch ---
            if model_class_name == 'SequentialTCNModel':
                model_type = "tcn"
                # Проверка консистентности параметров TCN
                expected_channels = model_init_params['num_input_channels_model']
                if metadata['pca_used'] and pca is not None and hasattr(pca,'n_components_') and expected_channels != pca.n_components_: print(f"Warn TCN: Каналы({expected_channels})!=PCA({pca.n_components_})")
                elif not metadata['pca_used'] and expected_channels != metadata['original_num_base_features']: print(f"Warn TCN: Каналы({expected_channels})!=Orig({metadata['original_num_base_features']})")
                # Создаем экземпляр TCN
                loaded_model = SequentialTCNModel(**model_init_params).to(inference_device)

            elif model_class_name == 'LSTMModel':
                model_type = "lstm"
                 # Проверка консистентности параметров LSTM
                expected_input_size = model_init_params['input_size']
                if metadata['pca_used'] and pca is not None and hasattr(pca,'n_components_') and expected_input_size != pca.n_components_: print(f"Warn LSTM: Input({expected_input_size})!=PCA({pca.n_components_})")
                elif not metadata['pca_used'] and expected_input_size != metadata['original_num_base_features']: print(f"Warn LSTM: Input({expected_input_size})!=Orig({metadata['original_num_base_features']})")
                # Создаем экземпляр LSTM
                loaded_model = LSTMModel(**model_init_params).to(inference_device)

            else:
                raise ValueError(f"Неподдерживаемый класс модели в .pth файле: {model_class_name}")

            # Загружаем веса и переводим в режим оценки
            loaded_model.load_state_dict(model_state_dict)
            loaded_model.eval()
            print(f"Загружена модель {model_type.upper()}.")

    except Exception as e:
        print(f"Ошибка при загрузке/воссоздании модели {model_path}: {e}")
        traceback.print_exc(); return None, None

    # --- Подготовка данных и Предсказание ---
    if not required_columns: print("Ошибка: Не загружен список признаков."); return None, None
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols: print(f"Ошибка: В DataFrame отсутствуют колонки: {missing_cols}"); return None, None

    # Функция для предобработки данных (общая для user и all)
    def preprocess_data(X_flat_local, n_samples_local):
        """Предобрабатывает 'плоские' данные (Scaler, PCA, Reshape)."""
        X_processed = X_flat_local # Начинаем с исходных плоских данных

        # --- Логика Scaler и PCA ДО Reshape ---
        if metadata and metadata.get('pca_used'):
            # Если PCA использовался при обучении, scaler и pca ожидают 2D (N*L, C_orig)
            if pca is None: raise ValueError("PCA использовался, но не найден.")
            if scaler is None: raise ValueError("Scaler не найден, но PCA использовался.")

            # 1. Reshape в 2D для Scaler/PCA
            X_2d = X_processed.reshape(-1, metadata['original_num_base_features'])

            # 2. Применяем Scaler
            X_2d_scaled = scaler.transform(X_2d)

            # 3. Применяем PCA
            X_2d_pca = pca.transform(X_2d_scaled)
            n_components = X_2d_pca.shape[1]

            # На этом этапе данные готовы к финальному reshape
            X_ready_for_reshape = X_2d_pca
            final_C = n_components # Количество признаков/каналов после PCA

        else: # Без PCA, Scaler обучался на плоских данных (N, C_orig*L)
            if scaler:
                # 1. Применяем Scaler к плоским данным
                X_flat_scaled = scaler.transform(X_processed)
            else:
                 X_flat_scaled = X_processed # Если scaler не сохранен

            # Данные готовы к финальному reshape
            X_ready_for_reshape = X_flat_scaled
            final_C = metadata['original_num_base_features'] # Количество признаков = исходное

        # --- Финальный Reshape в формат модели (N, L, C) или (N, C, L) ---
        reshape_order = metadata.get('input_reshape_order', 'NCL') # TCN по умолчанию

        # Важно: reshape нужно делать из правильной формы
        if metadata and metadata.get('pca_used'):
            # X_ready_for_reshape имеет форму (N*L, C_pca)
            if reshape_order == 'NLC': # LSTM
                X_seq = X_ready_for_reshape.reshape(n_samples_local, metadata['sequence_length'], final_C)
            else: # TCN
                X_seq = X_ready_for_reshape.reshape(n_samples_local, final_C, metadata['sequence_length'])
        else:
            # X_ready_for_reshape имеет форму (N, C_orig*L)
            if reshape_order == 'NLC': # LSTM
                X_seq = X_ready_for_reshape.reshape(n_samples_local, metadata['sequence_length'], final_C)
            else: # TCN
                X_seq = X_ready_for_reshape.reshape(n_samples_local, final_C, metadata['sequence_length'])

        return X_seq

    # --- Предсказание для пользователя ---
    user_pred = None
    user_row = df[df["Полное имя пользователя"] == fio]
    if user_row.empty: print(f"Предупреждение: Пользователь {fio} не найден.")
    else:
        try:
            X_user_flat = user_row[required_columns].values # (1, C*L)
            n_user = X_user_flat.shape[0] # = 1

            if model_type == "sklearn":
                X_user_proc = X_user_flat
                if scaler: X_user_proc = scaler.transform(X_user_proc)
                if pca: X_user_proc = pca.transform(X_user_proc)
                user_pred = loaded_model.predict(X_user_proc)[0]
            elif model_type in ["tcn", "lstm"]:
                X_user_seq = preprocess_data(X_user_flat, n_user)
                X_user_tensor = torch.tensor(X_user_seq, dtype=torch.float32).to(inference_device)
                with torch.no_grad(): user_pred_tensor = loaded_model(X_user_tensor)
                user_pred = user_pred_tensor.cpu().numpy().flatten()[0]
        except Exception as e: print(f"Ошибка предсказания для {fio}: {e}"); traceback.print_exc(); user_pred = None

    # --- Предсказание для группы ---
    mean_pred = None
    try:
        X_all_flat = df[required_columns].values # (N, C*L)
        n_all = X_all_flat.shape[0]

        if model_type == "sklearn":
            X_all_proc = X_all_flat
            if scaler: X_all_proc = scaler.transform(X_all_proc)
            if pca: X_all_proc = pca.transform(X_all_proc)
            all_preds = loaded_model.predict(X_all_proc)
        elif model_type in ["tcn", "lstm"]:
            X_all_seq = preprocess_data(X_all_flat, n_all)
            X_all_tensor = torch.tensor(X_all_seq, dtype=torch.float32).to(inference_device)
            all_preds_list = []; batch_size = 128
            with torch.no_grad():
                for i in range(0, n_all, batch_size):
                    batch_X = X_all_tensor[i:min(i + batch_size, n_all)]
                    batch_pred = loaded_model(batch_X)
                    all_preds_list.append(batch_pred.cpu())
            all_preds_tensor = torch.cat(all_preds_list, dim=0)
            all_preds = all_preds_tensor.numpy().flatten()

        if len(all_preds) > 0: mean_pred = np.nanmean(all_preds) # nanmean на случай NaN
        else: mean_pred = None
    except Exception as e: print(f"Ошибка предсказания для группы: {e}"); traceback.print_exc(); mean_pred = None

    # --- Преобразование в стандартный float перед возвратом ---
    safe_user_pred = float(user_pred) if user_pred is not None and pd.notna(user_pred) else None
    safe_mean_pred = float(mean_pred) if mean_pred is not None and pd.notna(mean_pred) else None

    print(f"Результат для {fio}: user={safe_user_pred}, mean={safe_mean_pred}")
    return safe_user_pred, safe_mean_pred


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

@app.route('/get_available_courses', methods=['GET'])
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

@app.route('/course_data/<course_name>', methods=['GET'])
def get_course_data(course_name):
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

        # --- Определение текущего периода студента для предсказания ---
        periods_numeric = [25, 33, 50, 66]
        assessment_columns = [f'Оценка за интервал {p}%' for p in periods_numeric]
        current_period_for_prediction = 0  # Период, на основе которого делаем прогноз

        for i, p in enumerate(periods_numeric):
            col_name = assessment_columns[i]
            # Проверяем наличие оценки > 0 для определения пройденного периода
            if col_name in student_data and pd.notna(student_data[col_name]) and student_data[col_name] > 0:
                current_period_for_prediction = p  # Обновляем последний валидный период
            else:
                # Если оценка за 'p' отсутствует или <= 0, значит,
                # предыдущий период (если он был) - последний завершенный.
                # Модель, обученная на этом предыдущем периоде, и будет предсказывать.
                break  # Выходим из цикла, как только нашли первый "не пройденный" период

        # Если студент не завершил даже 25%, предсказать не можем
        if current_period_for_prediction == 0:
            print(f"Студент {fio} не завершил период 25% с оценкой > 0. Прогноз невозможен.")
            # Возвращаем текущие данные без прогноза
            user_pred = None
            mean_pred = None
            current_period_str = "0%"  # Указываем, что прогресса нет
        else:
            # --- Выбор пути к модели на основе ОПРЕДЕЛЕННОГО периода ---
            current_period_str = f"{current_period_for_prediction}%"
            model_path_to_use = BEST_MODEL_PATHS_PER_PERIOD.get(current_period_str)

            if not model_path_to_use:
                print(
                    f"Ошибка: Путь к модели для периода {current_period_str} не найден в BEST_MODEL_PATHS_PER_PERIOD.")
                return None  # Или вернуть ошибку 500

            if not os.path.exists(model_path_to_use):
                print(
                    f"Ошибка: Файл лучшей модели для периода {current_period_str} не найден по пути: {model_path_to_use}")
                return None

            print(f"Определен текущий период для прогноза: {current_period_str}")
            print(f"Используется модель: {model_path_to_use}")

            # --- Вызов функции инференса с выбранным путем ---
            user_pred, mean_pred = get_predictive_assessment(df, model_path_to_use, fio)

        response = OrderedDict()
        response["fio"] = fio
        response["course_name"] = course_name
        response["period"] = f'{current_period_for_prediction}%'

        # --- Изменения здесь ---
        # Получаем текущую оценку (оставляем как есть, если она уже нужного типа или None)
        current_assessment = student_data.get(f'Оценка за интервал {current_period_for_prediction}%')
        response["current_assessment"] = float(current_assessment) if pd.notna(
            current_assessment) else None  # Преобразуем, если не NaN/None

        # Получаем среднюю текущую оценку
        current_mean = df[f'Оценка за интервал {current_period_for_prediction}%'].mean()
        response["current_assessment_mean"] = float(current_mean) if pd.notna(
            current_mean) else None  # Преобразуем, если не NaN/None

        # Преобразуем предсказания в стандартный float, если они не None
        response["predictive_assessment"] = float(user_pred) if user_pred is not None else None
        response["predictive_assessment_mean"] = float(mean_pred) if mean_pred is not None else None
        # --- Конец изменений ---

        # --- Формирование данных для графиков ---
        metric_labels = [  # Основные метрики для графиков
            "Число просмотров модулей", "Число входов в курс", "Число просмотров своих ошибок",
            "Число просмотров полученных оценок", "Количество выполненных заданий",
            "Среднее время между заходами"]
        periods_for_charts = [p for p in periods_numeric if
                              p <= current_period_for_prediction] if current_period_for_prediction > 0 else []

        charts = []
        for label in metric_labels:
            student_points = []
            group_average_points = []
            for percent in periods_for_charts:
                student_val = student_data.get(f"{label} {percent}%")
                group_val = df[f"{label} {percent}%"].mean() if f"{label} {percent}%" in df else None
                student_points.append(
                    {"period": percent, "value": float(student_val) if pd.notna(student_val) else None})
                group_average_points.append(
                    {"period": percent, "value": float(group_val) if pd.notna(group_val) else None})

            chart = {"option": label, "student": student_points, "group_average": group_average_points}
            charts.append(chart)

        response["charts"] = charts

        # Возвращаем через Response, чтобы не сломать порядок
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )

    except Exception as e:
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)