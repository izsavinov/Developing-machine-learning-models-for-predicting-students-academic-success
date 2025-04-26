import pandas as pd
import numpy as np


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

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

class TCNModel(nn.Module):
    """
    Класс модели TCN для PyTorch.
    """
    def __init__(self, input_size, filters, kernel_size, dilation_rate, dropout_rate):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, filters, kernel_size, padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same', dilation=dilation_rate * 2)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)  # Адаптивный пулинг
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters, 64)  # Линейный слой после пулинга
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_and_train_tcn(df, all_features=True, use_pca=False):
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

        # Преобразование данных в формат для TCN (samples, timesteps, features)
        X_train_tcn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_tcn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # Динамический подбор количества фильтров в зависимости от количества признаков
        # num_features = X_train_scaled.shape[1]  # Количество признаков
        # filters_options = {
        #     range(1, 8): 32,       # Для малого количества признаков (1–7)
        #     range(8, 16): 64,      # Для среднего количества признаков (8–15)
        #     range(16, 25): 128     # Для большого количества признаков (16–24)
        # }
        # filters = next(filters_options[r] for r in filters_options if num_features in r)

        # Определение гиперпараметров для ручного перебора
        param_grid = {
            'filters': [32, 64, 128],
            'kernel_size': [3, 5, 7],  # Размер ядра свёртки
            'dilation_rate': [1, 2, 4],  # Коэффициент расширения
            'dropout_rate': [0.1, 0.2, 0.3],  # Dropout для предотвращения переобучения
            'learning_rate': [0.001, 0.01, 0.1],  # Скорость обучения
            'batch_size': [16, 32, 64],  # Размер батча
            'epochs': [30, 50]  # Количество эпох
        }

        # Ручной перебор гиперпараметров
        best_mse = float('inf')
        best_params = None
        best_model = None

        for filters in param_grid['filters']:
            for kernel_size in param_grid['kernel_size']:
                for dilation_rate in param_grid['dilation_rate']:
                    for dropout_rate in param_grid['dropout_rate']:
                        for learning_rate in param_grid['learning_rate']:
                            for batch_size in param_grid['batch_size']:
                                for epochs in param_grid['epochs']:
                                    print(f"Тестирование параметров: kernel_size={kernel_size}, "
                                          f"dilation_rate={dilation_rate}, dropout_rate={dropout_rate}, "
                                          f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

                                    # Создание модели
                                    input_size = X_train_tcn.shape[2]
                                    model = TCNModel(input_size=input_size, filters=filters,
                                                     kernel_size=kernel_size, dilation_rate=dilation_rate,
                                                     dropout_rate=dropout_rate)
                                    model.to(device)

                                    # Оптимизатор и функция потерь
                                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                    criterion = nn.MSELoss()

                                    # Подготовка данных для PyTorch
                                    X_train_tensor = torch.tensor(X_train_tcn, dtype=torch.float32).to(device)
                                    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
                                    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                                    # Обучение модели
                                    model.train()
                                    for epoch in range(epochs):
                                        for batch_X, batch_y in dataloader:
                                            optimizer.zero_grad()
                                            outputs = model(batch_X.transpose(1, 2))  # TCN ожидает (batch, channels, length)
                                            loss = criterion(outputs.squeeze(), batch_y)
                                            loss.backward()
                                            optimizer.step()

                                    # Оценка модели на тестовой выборке
                                    model.eval()
                                    with torch.no_grad():
                                        X_test_tensor = torch.tensor(X_test_tcn, dtype=torch.float32).to(device)
                                        y_test_pred = model(X_test_tensor.transpose(1, 2)).cpu().numpy().flatten()
                                    test_mse = mean_squared_error(y_test, y_test_pred)

                                    # Сохранение лучших параметров
                                    if test_mse < best_mse:
                                        best_mse = test_mse
                                        best_params = {
                                            'filters': filters,
                                            'kernel_size': kernel_size,
                                            'dilation_rate': dilation_rate,
                                            'dropout_rate': dropout_rate,
                                            'learning_rate': learning_rate,
                                            'batch_size': batch_size,
                                            'epochs': epochs
                                        }
                                        best_model = model

        # Оценка лучшей модели на тестовой выборке
        best_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_tcn, dtype=torch.float32).to(device)
            y_test_pred = best_model(X_test_tensor.transpose(1, 2)).cpu().numpy().flatten()
        test_mse = mean_squared_error(y_test, y_test_pred)
        rmse = test_mse ** 0.5
        r2 = r2_score(y_test, y_test_pred)

        # Ручная 10-кратная кросс-валидация
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kf.split(X_train_tcn):
            X_train_fold, X_val_fold = X_train_tcn[train_index], X_train_tcn[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Создание модели для фолда
            fold_model = TCNModel(input_size=input_size, filters=best_params['filters'],
                                  kernel_size=best_params['kernel_size'],
                                  dilation_rate=best_params['dilation_rate'],
                                  dropout_rate=best_params['dropout_rate'])
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
                    outputs = fold_model(batch_X.transpose(1, 2))
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

            # Оценка модели на валидационном фолде
            fold_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
                y_val_pred = fold_model(X_val_tensor.transpose(1, 2)).cpu().numpy().flatten()
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
    cols = ["Test RMSE", "R²", "CV MSE"]

    # Считаем среднее по этим столбцам
    mean_values = df[cols].mean()

    # Вывести результат
    print(mean_values)

results_df_tcn, results_tcn = build_and_train_tcn(df_cleaned, True)
output_mean_metrics(results_df_tcn)

results_df_tcn_pca, results_tcn_pca = build_and_train_tcn(df_cleaned, True, True)
output_mean_metrics(results_df_tcn_pca)