import pandas as pd
import numpy as np
import joblib


periods = ["25%", "33%", "50%", "66%"]

file_path = '../data/Общее_Агрегированные_данные_проценты.xlsx'

df = pd.read_excel(file_path)

# Анализ целевой переменной (оценка за интервалы)
target_column = "Оценка за интервал 100%"

# Вычисление квартилей
Q1 = df[target_column].quantile(0.25)  # Первый квартиль (25%)
Q3 = df[target_column].quantile(0.75)  # Третий квартиль (75%)
IQR = Q3 - Q1  # Межквартильный размах

# Определение границ для выбросов
lower_bound = Q1 - 1.5 * IQR  # Нижняя граница
upper_bound = Q3 + 1.5 * IQR  # Верхняя граница
# Фильтрация данных (удаление выбросов)
df_cleaned = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]

# Сохранение очищенной целевой переменной
Y_clean = df_cleaned[target_column]

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed

class LSTMModel(nn.Module):
    """
    Класс модели LSTM для PyTorch.
    """
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Берем только последний временной шаг
        dropout_out = self.dropout(lstm_out)
        output = self.fc(dropout_out)
        return output

def build_and_train_lstm_(df, all_features=True, use_pca=False):

    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Результаты для всех периодов
    results = []

    # Цикл по всем периодам
    for period in periods:
        # Выбор признаков в зависимости от параметра all_features
        if all_features:
            features = []
            for p in periods[:periods.index(period) + 1]:  # Включаем текущий и предыдущие периоды
                features.extend([col for col in df.columns if col.endswith(p) and df[col].dtype != 'object'])
        else:
            features = [col for col in df.columns if col.endswith(period) and df[col].dtype != 'object']

        # Формирование признаков и целевой переменной
        X = df[features]
        y = df[target_column]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Применение PCA (если указано)
        if use_pca:
            pca = PCA(n_components=0.9)  # Сохраняем 90% дисперсии
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Масштабирование данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Преобразование данных в формат для LSTM (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # Определение гиперпараметров для ручного перебора
        param_grid = {
            'hidden_size': [32, 64, 128],  # Количество нейронов в LSTM-слое
            'dropout_rate': [0.1, 0.2, 0.3],  # Dropout для предотвращения переобучения
            'learning_rate': [0.001, 0.01, 0.1],  # Скорость обучения
            'batch_size': [16, 32, 64],  # Размер батча
            'epochs': [30, 50]  # Количество эпох
        }

        # Ручной перебор гиперпараметров
        best_mse = float('inf')
        best_params = None
        best_model = None

        for hidden_size in param_grid['hidden_size']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:
                        for epochs in param_grid['epochs']:
                            print(f"Тестирование параметров: hidden_size={hidden_size}, "
                                  f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                                  f"batch_size={batch_size}, epochs={epochs}")

                            # Создание модели
                            input_size = X_train_lstm.shape[2]
                            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
                            model.to(device)

                            # Оптимизатор и функция потерь
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                            criterion = nn.MSELoss()

                            # Подготовка данных для PyTorch
                            X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32).to(device)
                            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
                            dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                            # Обучение модели
                            model.train()
                            for epoch in range(epochs):
                                for batch_X, batch_y in dataloader:
                                    optimizer.zero_grad()
                                    outputs = model(batch_X)
                                    loss = criterion(outputs.squeeze(), batch_y)
                                    loss.backward()
                                    optimizer.step()

                            # Оценка модели на тестовой выборке
                            model.eval()
                            with torch.no_grad():
                                X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
                                y_test_pred = model(X_test_tensor).cpu().numpy().flatten()
                            test_mse = mean_squared_error(y_test, y_test_pred)

                            # Сохранение лучших параметров
                            if test_mse < best_mse:
                                best_mse = test_mse
                                best_params = {
                                    'hidden_size': hidden_size,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs
                                }
                                best_model = model

        # Оценка лучшей модели на тестовой выборке
        best_model.eval()

        # Сохранение лучшей модели в файл
        model_filename = f"best_knn_model_period_{period}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"Модель сохранена в файл: {model_filename}\n")

        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
            y_test_pred = best_model(X_test_tensor).cpu().numpy().flatten()
        test_mse = mean_squared_error(y_test, y_test_pred)
        rmse = test_mse ** 0.5
        r2 = r2_score(y_test, y_test_pred)

        # Ручная 10-кратная кросс-валидация
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kf.split(X_train_lstm):
            X_train_fold, X_val_fold = X_train_lstm[train_index], X_train_lstm[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Создание модели для фолда
            fold_model = LSTMModel(input_size=input_size, hidden_size=best_params['hidden_size'], dropout_rate=best_params['dropout_rate'])
            fold_model.to(device)

            # Подготовка данных для PyTorch
            X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
            y_train_fold_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).to(device)
            fold_dataset = torch.utils.data.TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
            fold_dataloader = torch.utils.data.DataLoader(fold_dataset, batch_size=best_params['batch_size'], shuffle=True)

            # Обучение модели на фолде
            optimizer = optim.Adam(fold_model.parameters(), lr=best_params['learning_rate'])
            fold_model.train()
            for epoch in range(best_params['epochs']):
                for batch_X, batch_y in fold_dataloader:
                    optimizer.zero_grad()
                    outputs = fold_model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

            # Оценка модели на валидационном фолде
            fold_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
                y_val_pred = fold_model(X_val_tensor).cpu().numpy().flatten()
            fold_mse = mean_squared_error(y_val_fold, y_val_pred)
            cv_scores.append(fold_mse)

        # Средние метрики кросс-валидации
        cv_mse = np.mean(cv_scores)
        cv_rmse = np.sqrt(cv_mse)

        # Сохранение результатов
        results.append({
            'Period': period,
            'Best Parameters': best_params,
            'Test MSE': test_mse,
            'Test RMSE': rmse,
            'R²': r2,
            'CV MSE': cv_mse,
            'CV RMSE': cv_rmse
        })

        print(f"Результаты для периода {period}:")
        print(f"Лучшие параметры: {best_params}")
        print(f"Test MSE={test_mse:.2f}, Test RMSE={rmse:.2f}, R²={r2:.2f}")
        print(f"CV MSE={cv_mse:.2f}, CV RMSE={cv_rmse:.2f}")
        print("-" * 50)

    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df, results

def output_mean_metrics(data):
    df = pd.DataFrame(data)

    # Приведение всех значений к типу float на случай, если что-то пошло не так с типами
    df = pd.DataFrame(data)

    # Указываем нужные столбцы
    cols = ["Test RMSE", "R²", "CV RMSE", "AIC", "BIC"]

    # Считаем среднее по этим столбцам
    mean_values = df[cols].mean()

    # Вывести результат
    print(mean_values)

def process_period(period, df, all_features, use_pca, target_column, device):
    """
    Обработка одного периода.
    """
    # Выбор признаков в зависимости от параметра all_features
    if all_features:
        features = []
        for p in periods[:periods.index(period) + 1]:  # Включаем текущий и предыдущие периоды
            features.extend([col for col in df.columns if col.endswith(p) and df[col].dtype != 'object'])
    else:
        features = [col for col in df.columns if col.endswith(period) and df[col].dtype != 'object']

    # Формирование признаков и целевой переменной
    X = df[features]
    y = df[target_column]

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Применение PCA (если указано)
    if use_pca:
        pca = PCA(n_components=0.9)  # Сохраняем 90% дисперсии
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Преобразование данных в формат для LSTM (samples, timesteps, features)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Определение гиперпараметров для ручного перебора
    param_grid = {
        'hidden_size': [32, 64, 128],  # Количество нейронов в LSTM-слое
        'dropout_rate': [0.1, 0.2, 0.3],  # Dropout для предотвращения переобучения
        'learning_rate': [0.001, 0.01, 0.1],  # Скорость обучения
        'batch_size': [16, 32, 64],  # Размер батча
        'epochs': [30, 50]  # Количество эпох
    }

    # Ручной перебор гиперпараметров
    best_mse = float('inf')
    best_params = None
    best_model = None

    for hidden_size in param_grid['hidden_size']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        print(f"[Период {period}] Тестирование параметров: hidden_size={hidden_size}, "
                              f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                              f"batch_size={batch_size}, epochs={epochs}")

                        # Создание модели
                        input_size = X_train_lstm.shape[2]
                        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
                        model.to(device)

                        # Оптимизатор и функция потерь
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = nn.MSELoss()

                        # Подготовка данных для PyTorch
                        X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32).to(device)
                        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
                        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        # Обучение модели
                        model.train()
                        for epoch in range(epochs):
                            for batch_X, batch_y in dataloader:
                                optimizer.zero_grad()
                                outputs = model(batch_X)
                                loss = criterion(outputs.squeeze(), batch_y)
                                loss.backward()
                                optimizer.step()

                        # Оценка модели на тестовой выборке
                        model.eval()
                        with torch.no_grad():
                            X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
                            y_test_pred = model(X_test_tensor).cpu().numpy().flatten()
                        test_mse = mean_squared_error(y_test, y_test_pred)

                        # Сохранение лучших параметров
                        if test_mse < best_mse:
                            best_mse = test_mse
                            best_params = {
                                'hidden_size': hidden_size,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }
                            best_model = model

    # Оценка лучшей модели на тестовой выборке
    best_model.eval()

    # Сохранение лучшей модели в файл
    model_filename = f"best_lstm_model_period_{period}.joblib"
    joblib.dump(best_model, model_filename)
    print(f"[Период {period}] Модель сохранена в файл: {model_filename}\n")

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
        y_test_pred = best_model(X_test_tensor).cpu().numpy().flatten()
    test_mse = mean_squared_error(y_test, y_test_pred)
    rmse = test_mse ** 0.5
    r2 = r2_score(y_test, y_test_pred)

    # Ручная 10-кратная кросс-валидация
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []

    for train_index, val_index in kf.split(X_train_lstm):
        X_train_fold, X_val_fold = X_train_lstm[train_index], X_train_lstm[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Создание модели для фолда
        fold_model = LSTMModel(input_size=input_size, hidden_size=best_params['hidden_size'], dropout_rate=best_params['dropout_rate'])
        fold_model.to(device)

        # Подготовка данных для PyTorch
        X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_fold_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).to(device)
        fold_dataset = torch.utils.data.TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
        fold_dataloader = torch.utils.data.DataLoader(fold_dataset, batch_size=best_params['batch_size'], shuffle=True)

        # Обучение модели на фолде
        optimizer = optim.Adam(fold_model.parameters(), lr=best_params['learning_rate'])
        fold_model.train()
        for epoch in range(best_params['epochs']):
            for batch_X, batch_y in fold_dataloader:
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

        # Оценка модели на валидационном фолде
        fold_model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
            y_val_pred = fold_model(X_val_tensor).cpu().numpy().flatten()
        fold_mse = mean_squared_error(y_val_fold, y_val_pred)
        cv_scores.append(fold_mse)

    # Средние метрики кросс-валидации
    cv_mse = np.mean(cv_scores)
    cv_rmse = np.sqrt(cv_mse)

    # Возвращаем результаты для периода
    return {
        'Period': period,
        'Best Parameters': best_params,
        'Test MSE': test_mse,
        'Test RMSE': rmse,
        'R²': r2,
        'CV MSE': cv_mse,
        'CV RMSE': cv_rmse
    }

def build_and_train_lstm(df, all_features=True, use_pca=False):
    """
    Главная функция для обучения LSTM-моделей.
    """
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Параллельная обработка всех периодов
    results = Parallel(n_jobs=-1)(
        delayed(process_period)(period, df, all_features, use_pca, target_column, device)
        for period in periods
    )

    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df, results

results_df_LSTM, results__LSTM = build_and_train_lstm(df_cleaned, True)
output_mean_metrics(results_df_LSTM)

results_df_LSTM_pca, results__LSTM_pca = build_and_train_lstm(df_cleaned, True, True)
output_mean_metrics(results_df_LSTM_pca)