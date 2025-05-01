import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# <<< Добавляем KFold и Subset >>>
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# <<< Добавляем Subset >>>
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
import copy
import pandas as pd
import optuna
import time
import warnings
import os
import traceback
from torchinfo import summary
import sys  # Для проверки импорта torchviz в finally

warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Настройки Pandas ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- Константы и Загрузка данных ---
periods = ["25%", "33%", "50%", "66%"]
file_path = '../data/Общее_Агрегированные_данные_проценты.xlsx'
base_feature_names = [
    "Число входов в курс", "Число просмотров модулей", "Число просмотров своих ошибок",
    "Число просмотров полученных оценок", "Количество выполненных заданий",
    "Среднее время между заходами", "Оценка за интервал"]
num_base_features = len(base_feature_names)
print(f"Определено {num_base_features} базовых признаков: {base_feature_names}")
try:
    df_original = pd.read_excel(file_path)
    print(f"Данные успешно загружены. Форма: {df_original.shape}")
    all_expected_cols = [f"{base} {p}" for p in periods for base in base_feature_names];
    missing_cols = [col for col in all_expected_cols if col not in df_original.columns]
    if missing_cols:
        print(f"ОШИБКА: Отсутствуют колонки: {missing_cols}.");
        exit()
    else:
        print("Все ожидаемые колонки присутствуют.")
except FileNotFoundError:
    print(f"Ошибка: Файл не найден: {file_path}. Создание dummy данных.")
    num_students = 200;
    data = {'ID_студента': range(num_students)}
    for p in periods:
        for base_name in base_feature_names: data[f"{base_name} {p}"] = np.random.rand(num_students) * 100
    data["Оценка за интервал 100%"] = np.random.rand(num_students) * 70 + 30;
    df_original = pd.DataFrame(data);
    print(f"Dummy DataFrame создан. Форма: {df_original.shape}")

target_column = "Оценка за интервал 100%"
# --- Очистка выбросов ---
if target_column in df_original.columns:
    if pd.api.types.is_numeric_dtype(df_original[target_column]):
        Q1 = df_original[target_column].quantile(0.25);
        Q3 = df_original[target_column].quantile(0.75);
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR;
        upper_bound = Q3 + 1.5 * IQR;
        df_cleaned = df_original[
            (df_original[target_column] >= lower_bound) & (df_original[target_column] <= upper_bound)].copy()
        print(f"Очистка выбросов: {df_cleaned.shape} (удалено {df_original.shape[0] - df_cleaned.shape[0]})")
    else:
        print("Предупреждение: Целевая колонка не числовая.");
        df_cleaned = df_original.copy()
else:
    print(f"Ошибка: Целевая колонка '{target_column}' не найдена.");
    exit()
# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
print(f"Устройство: {device}")


# --- Вспомогательные функции ---
def get_activation_fn(activation_name):
    """Возвращает класс функции активации PyTorch по имени."""
    if activation_name == 'ReLU':
        return nn.ReLU
    elif activation_name == 'GELU':
        return nn.GELU
    elif activation_name == 'LeakyReLU':
        return nn.LeakyReLU
    else:
        raise ValueError(f"Неизвестная функция активации: {activation_name}")


def get_optimizer(optimizer_name):
    """Возвращает класс оптимизатора PyTorch по имени."""
    if optimizer_name == 'Adam':
        return optim.Adam
    elif optimizer_name == 'AdamW':
        return optim.AdamW
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")


# --- Генерация каналов для TCN ---
def generate_channels(trial_or_params, num_layers, num_input_channels_model):
    """Генерирует список с количеством каналов для слоев TCN."""
    channels = []
    # Проверяем, работаем ли мы с триалом Optuna или словарем параметров
    is_trial = isinstance(trial_or_params, optuna.trial.Trial)

    # Определяем ключи для параметров
    pattern_key = 'channel_pattern'
    factor_inc_key = 'channel_inc_factor'
    factor_dec_key = 'channel_dec_factor'
    first_channel_key = 'first_channel'

    # Получаем паттерн (или предлагаем, если это триал)
    pattern_options = ['constant', 'increase', 'decrease', 'pyramid']
    pattern = trial_or_params.suggest_categorical(pattern_key, pattern_options) if is_trial else trial_or_params.get(
        pattern_key, 'constant')

    # Определяем диапазон и шаг для первого слоя каналов
    min_ch = max(8, num_input_channels_model)  # Не меньше 8 и не меньше входных каналов
    max_ch = max(min_ch, min(128, num_input_channels_model * 4))  # Ограничиваем сверху
    step = max(8, (max_ch - min_ch) // 4) if max_ch > min_ch else 8
    first_channel_options = list(range(min_ch, max_ch + 1, step))
    if not first_channel_options:  # Если диапазон слишком мал
        first_channel_options = [min_ch]

    # Получаем количество каналов первого слоя (или предлагаем, если триал)
    default_first_channel = max(16, num_input_channels_model)
    first_channel = trial_or_params.suggest_categorical(first_channel_key,
                                                        first_channel_options) if is_trial else trial_or_params.get(
        first_channel_key, default_first_channel)

    # Генерируем список каналов в соответствии с паттерном
    if pattern == 'constant':
        channels = [first_channel] * num_layers
    elif pattern == 'increase':
        factor = trial_or_params.suggest_float(factor_inc_key, 1.1, 2.0) if is_trial else trial_or_params.get(
            factor_inc_key, 1.5)
        current_channel = float(first_channel)  # Начинаем с float для точности умножения
        for _ in range(num_layers):
            channels.append(max(8, min(512, int(round(current_channel)))))  # Округляем и ограничиваем
            current_channel *= factor
    elif pattern == 'decrease':
        factor = trial_or_params.suggest_float(factor_dec_key, 0.5, 0.9) if is_trial else trial_or_params.get(
            factor_dec_key, 0.7)
        current_channel = float(first_channel)
        for _ in range(num_layers):
            channels.append(max(8, min(512, int(round(current_channel)))))
            current_channel *= factor
    elif pattern == 'pyramid':
        peak_layer = num_layers // 2  # Слой с максимальным количеством каналов (примерно середина)
        inc_factor = trial_or_params.suggest_float(factor_inc_key, 1.1, 2.0) if is_trial else trial_or_params.get(
            factor_inc_key, 1.5)
        dec_factor = trial_or_params.suggest_float(factor_dec_key, 0.5, 0.9) if is_trial else trial_or_params.get(
            factor_dec_key, 0.7)
        current_channel = float(first_channel)
        # Фаза роста
        for i in range(peak_layer + 1):
            channels.append(max(8, min(512, int(round(current_channel)))))
            if i < peak_layer:  # Увеличиваем до пикового слоя
                current_channel *= inc_factor
        # Фаза спада (начинаем с current_channel после пика, умноженного на dec_factor)
        current_channel *= dec_factor
        for _ in range(peak_layer + 1, num_layers):
            channels.append(max(8, min(512, int(round(current_channel)))))
            current_channel *= dec_factor
        # Убедимся, что итоговая длина списка равна num_layers
        channels = channels[:num_layers]

    # Запасной вариант, если список каналов пуст
    if not channels and num_layers > 0:
        channels = [max(8, min(512, first_channel))] * num_layers

    # Финальная проверка минимального значения
    channels = [max(8, c) for c in channels]
    return channels


# --- Каузальный Сверточный Блок для TCN ---
class CausalConv1dBlock(nn.Module):
    """Один блок каузальной свертки для TCN."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation_fn, dropout_rate):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation  # Расчет каузального паддинга
        Activation = get_activation_fn(activation_fn)
        self.pad = nn.ConstantPad1d((self.causal_padding, 0), 0)  # Паддинг только слева
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)  # Нормализация
        self.activation = Activation()  # Функция активации
        self.dropout = nn.Dropout(dropout_rate)  # Dropout для регуляризации
        # Опциональная инициализация весов
        # nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        """Прямой проход через блок."""
        out = self.pad(x)  # Добавляем паддинг
        out = self.conv(out)  # Свертка
        out = self.norm(out)  # Нормализация
        out = self.activation(out)  # Активация
        out = self.dropout(out)  # Dropout
        return out


# --- Модель TCN с Каузальными Свертками ---
class SequentialTCNModel(nn.Module):
    """Модель TCN, состоящая из последовательности CausalConv1dBlock."""

    def __init__(self, num_input_channels_model, output_size, num_channels, kernel_size, dropout_rate, activation_fn,
                 dilation_base=2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = num_input_channels_model  # Количество каналов на входе первого блока

        # Создаем последовательность сверточных блоков
        for i in range(num_levels):
            dilation_size = dilation_base ** i  # Экспоненциальный рост дилатации
            out_channels = num_channels[i]  # Количество каналов на выходе текущего блока
            layers.append(
                CausalConv1dBlock(
                    in_channels, out_channels, kernel_size,
                    dilation=dilation_size, activation_fn=activation_fn,
                    dropout_rate=dropout_rate
                )
            )
            in_channels = out_channels  # Выходные каналы становятся входными для следующего слоя

        self.network = nn.Sequential(*layers)  # Объединяем блоки в последовательную сеть

        # Выходной слой
        self.output_channels = num_channels[
            -1] if num_channels else num_input_channels_model  # Кол-во каналов после последнего блока
        self.output_layer = nn.Linear(self.output_channels, output_size)  # Полносвязный слой для регрессии

    def forward(self, x):
        """
        Прямой проход данных через всю TCN.
        Args:
            x (torch.Tensor): Входной тензор формы (N, C, L).
        Returns:
            torch.Tensor: Выходной тензор формы (N, output_size).
        """
        out = self.network(x)  # Пропускаем через сверточные блоки -> (N, C_out_last, L)
        # Используем выход только последнего временного шага
        out = out[:, :, -1]  # -> (N, C_out_last)
        # Подаем на выходной слой
        out = self.output_layer(out)  # -> (N, output_size)
        # Проверки на NaN (можно убрать для производительности)
        # if torch.isnan(out).any(): print("Warning: NaN detected after output layer!")
        return out


# --- Функция обучения одного фолда или финальной модели (Общая) ---
# (Код функции train_single_fold без изменений, как в предыдущем ответе для LSTM)
def train_single_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, model_class, model_params, train_params,
                      device):
    """Обучает модель на одном фолде с ранней остановкой."""
    model = model_class(**model_params).to(device)
    optimizer_class = get_optimizer(train_params['optimizer_name'])
    optimizer = optimizer_class(model.parameters(), lr=train_params['learning_rate'],
                                weight_decay=train_params['weight_decay'])
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
    best_val_loss = float('inf');
    best_model_wts = copy.deepcopy(model.state_dict());
    epochs_no_improve = 0
    patience = train_params.get('patience', 20);
    epochs = train_params['epochs']
    train_dataset = TensorDataset(X_train_fold, torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,
                              pin_memory=device.type == 'cuda')
    val_dataset = TensorDataset(X_val_fold, torch.tensor(y_val_fold.values, dtype=torch.float32).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], pin_memory=device.type == 'cuda')
    scheduler = None
    if train_params.get('final_training', False) and train_params.get('scheduler_name'):
        scheduler_name = train_params['scheduler_name']
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2,
                                                             verbose=False)
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    for epoch in range(epochs):
        model.train();
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device);
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X);
            loss = criterion(outputs, batch_y)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward();
            optimizer.step();
            epoch_train_loss += loss.item() * batch_X.size(0)
        model.eval();
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_outputs = model(batch_X_val);
                val_loss = criterion(val_outputs, batch_y_val)
                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    epoch_val_loss += val_loss.item() * batch_X_val.size(0)
                else:
                    epoch_val_loss += 1e10 * batch_X_val.size(0)
        avg_train_loss = epoch_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        avg_val_loss = epoch_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('inf')
        train_rmse = np.sqrt(max(0, avg_train_loss));
        val_rmse = np.sqrt(max(0, avg_val_loss)) if not np.isinf(avg_val_loss) else float('inf')
        history['train_loss'].append(avg_train_loss);
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(train_rmse);
        history['val_rmse'].append(val_rmse)
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        if train_params.get('final_training', False) and ((epoch + 1) % 25 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]['lr'];
            print(
                f'    Эпоха {epoch + 1:3d}/{epochs}, Train RMSE: {train_rmse:.5f}, Val RMSE (Test): {val_rmse:.5f}, LR: {current_lr:.6f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss;
            best_model_wts = copy.deepcopy(model.state_dict());
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if train_params.get('final_training', False): print(f'    Ранняя остановка на эпохе {epoch + 1}.')
                break
        if np.isinf(avg_val_loss) or np.isnan(avg_val_loss):
            if train_params.get('final_training', False): print(
                f"    Остановка из-за некорр. Val Loss на эпохе {epoch + 1}.")
            break
    model.load_state_dict(best_model_wts)
    return model, history, best_val_loss


# --- Функция оценки (общая) ---
# (Код функции evaluate_model без изменений, как в предыдущих ответах)
def evaluate_model(model, X_test_tensor, y_test_series, device):
    """Оценивает обученную модель на тестовых данных."""
    model.eval();
    y_pred_list = [];
    y_true_list = y_test_series.values
    test_dataset = TensorDataset(X_test_tensor);
    test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=device.type == 'cuda')
    with torch.no_grad():
        for batch_X_tuple in test_loader: batch_X = batch_X_tuple[0].to(device, non_blocking=True);outputs = model(
            batch_X).cpu().numpy().flatten();y_pred_list.extend(outputs)
    y_pred = np.array(y_pred_list)
    if len(y_pred) != len(y_true_list): print(
        f"Warn: y_pred({len(y_pred)}) vs y_test({len(y_true_list)})");min_len = min(len(y_pred),
                                                                                    len(y_true_list));y_pred = y_pred[
                                                                                                               :min_len];y_true_list = y_true_list[
                                                                                                                                       :min_len]
    if np.isnan(y_pred).any() or np.isinf(y_pred).any(): print(
        "Warn: NaN/Inf preds!");return np.nan, np.nan, np.nan, y_pred
    mse = mean_squared_error(y_true_list, y_pred);
    rmse = np.sqrt(max(0, mse))
    if np.var(y_true_list) < 1e-9:
        r2 = np.nan if mse > 1e-9 else 1.0
    else:
        r2 = r2_score(y_true_list, y_pred)
    return mse, rmse, r2, y_pred


# --- Optuna objective для TCN с Кросс-Валидацией ---
def objective_tcn_cv(trial, X_train_val_tensor, y_train_val_series, num_input_channels_model, sequence_length, device,
                     n_splits=5):
    """Целевая функция Optuna для TCN, использующая K-Fold CV."""

    # --- Параметры архитектуры TCN ---
    num_layers = trial.suggest_int('num_layers', 2, 5)  # Немного уменьшим макс. слоев для CV
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 5])  # Уменьшим варианты
    dilation_base = trial.suggest_categorical('dilation_base', [1, 2])  # Уменьшим варианты
    num_channels = generate_channels(trial, num_layers, num_input_channels_model)
    activation_fn_name = trial.suggest_categorical('activation', ['ReLU', 'GELU'])  # Уменьшим варианты
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4, step=0.1)  # Уменьшим dropout

    # --- Параметры обучения ---
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    patience_value = trial.suggest_int('patience', 7, 15)
    epochs = 100  # Меньше эпох для CV

    # Собираем параметры для модели и обучения
    model_params = {
        'num_input_channels_model': num_input_channels_model, 'output_size': 1,
        'num_channels': num_channels, 'kernel_size': kernel_size,
        'dropout_rate': dropout_rate, 'activation_fn': activation_fn_name,
        'dilation_base': dilation_base}
    train_params = {
        'optimizer_name': optimizer_name, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'epochs': epochs, 'batch_size': batch_size, 'patience': patience_value,
        'final_training': False  # Это НЕ финальное обучение
    }

    # --- K-Fold Кросс-Валидация ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=trial.number)
    fold_val_losses = []
    indices = np.arange(len(X_train_val_tensor))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # print(f"  Триал {trial.number}, Фолд {fold+1}/{n_splits}") # Отладка
        X_train_fold = X_train_val_tensor[train_idx]
        y_train_fold = y_train_val_series.iloc[train_idx]
        X_val_fold = X_train_val_tensor[val_idx]
        y_val_fold = y_train_val_series.iloc[val_idx]

        try:
            # Обучаем модель на фолде
            # Игнорируем возвращаемую модель и историю, нужен только лосс
            _, _, best_fold_val_loss = train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                SequentialTCNModel, model_params, train_params, device)  # <<< Используем SequentialTCNModel

            if not np.isnan(best_fold_val_loss) and np.isfinite(best_fold_val_loss):
                fold_val_losses.append(best_fold_val_loss)
            else:
                print(
                    f"WARNING: Некорректный val_loss ({best_fold_val_loss}) TCN триал {trial.number}, фолд {fold + 1}.")
                return float('inf')  # Считаем триал неудачным

            # Прунинг внутри CV
            trial.report(np.mean(fold_val_losses) if fold_val_losses else float('inf'), step=fold)
            if trial.should_prune(): raise optuna.TrialPruned()

        except optuna.TrialPruned as e:
            raise e
        except Exception as e:
            print(f"Ошибка TCN фолда {fold + 1} триал {trial.number}: {e}");
            traceback.print_exc();
            return float('inf')

    # Возвращаем средний RMSE по фолдам
    avg_cv_loss = np.mean(fold_val_losses) if fold_val_losses else float('inf')
    avg_cv_rmse = np.sqrt(max(0, avg_cv_loss)) if not np.isinf(avg_cv_loss) else float('inf')

    print(f"  Триал TCN {trial.number} завершен. Средний CV RMSE: {avg_cv_rmse:.5f}")  # Отладка
    return avg_cv_rmse


# --- Основная функция обработки периода для TCN с CV в Optuna ---
def process_period_tcn_cv(period_index, df_clean, base_feature_names, device, n_trials=50, use_pca=False,
                          pca_variance=0.95, models_save_dir="saved_tcn_models_cv", n_splits_optuna=5):
    """Выполняет цикл обработки для TCN с K-Fold CV в Optuna."""
    current_period_label = periods[period_index];
    periods_to_include = periods[:period_index + 1]
    sequence_length = len(periods_to_include);
    original_num_base_features = len(base_feature_names)
    pca_info = f"с PCA (var={pca_variance * 100:.0f}%)" if use_pca else "без PCA"
    print(
        f"\n===== Запуск Optuna TCN (CV={n_splits_optuna}) до {current_period_label} ({n_trials} тр, {pca_info}, L={sequence_length}, C_orig={original_num_base_features}) =====")
    start_time_period = time.time()

    # 1. Формирование колонок
    # ... (код без изменений) ...
    feature_columns_ordered = []
    for p in periods_to_include:
        for base_name in base_feature_names:
            col_name = f"{base_name} {p}";
            if col_name in df_clean.columns:
                feature_columns_ordered.append(col_name)
            else:
                print(f"ОШИБКА: Колонка {col_name} не найдена.");
                return None
    if len(feature_columns_ordered) != original_num_base_features * sequence_length:
        print(f"ОШИБКА: Неверное кол-во признаков.");
        return None

    print(feature_columns_ordered)
    X_flat_df = df_clean[feature_columns_ordered];
    y = df_clean[target_column]

    # 2. Разделение данных на Train+Val и Test
    X_train_val_flat_df, X_test_flat_df, y_train_val, y_test = train_test_split(X_flat_df, y, test_size=0.2,
                                                                                random_state=42)
    print(f"Размеры плоских данных: Train+Val={X_train_val_flat_df.shape}, Test={X_test_flat_df.shape}")
    if X_train_val_flat_df.shape[0] < n_splits_optuna * 2:
        print(
            f"Предупреждение: Недостаточно данных ({X_train_val_flat_df.shape[0]}) для {n_splits_optuna}-Fold CV. Пропуск периода.")
        return None

    # 3. Подготовка данных Train+Val для Optuna (Scaler, PCA, Reshape в (N, C, L))
    final_scaler = StandardScaler();
    final_pca_model = None
    num_input_channels_model = original_num_base_features  # C для TCN
    n_train_val = X_train_val_flat_df.shape[0]
    try:
        if use_pca:
            print(f"Применение Scaler+PCA (обучение на Train+Val 2D)...")
            X_train_val_2d = X_train_val_flat_df.values.reshape(-1, original_num_base_features)
            X_train_val_2d_scaled = final_scaler.fit_transform(X_train_val_2d)
            final_pca_model = PCA(n_components=pca_variance, random_state=42).fit(X_train_val_2d_scaled)
            X_train_val_2d_pca = final_pca_model.transform(X_train_val_2d_scaled)
            n_components = X_train_val_2d_pca.shape[1];
            num_input_channels_model = n_components
            print(f"PCA: {n_components} комп. (на Train+Val).")
            if n_components == 0: raise ValueError("PCA 0 комп. (на Train+Val).")
            X_train_val_seq = X_train_val_2d_pca.reshape(n_train_val, n_components, sequence_length)  # (N, C, L)
        else:
            print("Применение Scaler (обучение на Train+Val плоских)...")
            X_train_val_flat_scaled = final_scaler.fit_transform(X_train_val_flat_df)
            X_train_val_seq = X_train_val_flat_scaled.reshape(n_train_val, original_num_base_features,
                                                              sequence_length)  # (N, C, L)
    except ValueError as e:
        print(f"Ошибка подготовки Train+Val данных: {e}");
        traceback.print_exc();
        return None
    X_train_val_tensor = torch.tensor(X_train_val_seq, dtype=torch.float32);
    y_train_val_series = y_train_val
    print(f"Форма данных для Optuna CV (TCN): {X_train_val_tensor.shape}")

    # 4. Запуск Optuna для TCN с Кросс-Валидацией
    print(f"Запуск Optuna TCN с {n_splits_optuna}-Fold CV (Input C={num_input_channels_model}, L={sequence_length})...")
    study_tcn_cv = optuna.create_study(direction="minimize",
                                       pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3,
                                                                          interval_steps=1))
    try:
        study_tcn_cv.optimize(
            lambda trial: objective_tcn_cv(trial, X_train_val_tensor, y_train_val_series, num_input_channels_model,
                                           sequence_length, device, n_splits=n_splits_optuna),
            n_trials=n_trials, timeout=10800)
    except Exception as e:
        print(f"Критическая ошибка Optuna TCN CV: {e}");
        traceback.print_exc();
    try:
        best_trial_tcn_cv = study_tcn_cv.best_trial
    except ValueError:
        print("Optuna TCN CV не нашла лучший триал.");
        return None
    best_params_tcn_cv = best_trial_tcn_cv.params;
    best_cv_rmse = best_trial_tcn_cv.value
    # Добавляем параметры обучения
    best_params_tcn_cv['learning_rate'] = best_trial_tcn_cv.params.get('learning_rate')
    best_params_tcn_cv['optimizer_name'] = best_trial_tcn_cv.params.get('optimizer')
    best_params_tcn_cv['weight_decay'] = best_trial_tcn_cv.params.get('weight_decay')
    best_params_tcn_cv['batch_size'] = best_trial_tcn_cv.params.get('batch_size')
    best_params_tcn_cv['actual_input_channels'] = num_input_channels_model

    print(f"\nOptuna TCN CV завершена. Лучший средний CV RMSE: {best_cv_rmse:.5f}")
    print(f"Лучшие параметры TCN (для {num_input_channels_model} вх. каналов):");
    [print(f"  {k}: {v}") for k, v in best_params_tcn_cv.items()]

    # 5. Обучение Финальной TCN Модели на ВСЕХ Train+Val данных
    print(f"\nОбучение финальной TCN модели для {current_period_label} на ВСЕХ Train+Val данных...")
    # --- Подготовка Тестовых данных ---
    n_test = X_test_flat_df.shape[0]
    final_num_input_channels_model = num_input_channels_model  # Input size уже определен
    try:
        if use_pca:
            X_test_2d = X_test_flat_df.values.reshape(-1, original_num_base_features)
            X_test_2d_scaled = final_scaler.transform(X_test_2d)  # Используем final_scaler
            X_test_2d_pca = final_pca_model.transform(X_test_2d_scaled)  # Используем final_pca_model
            X_test_seq_final = X_test_2d_pca.reshape(n_test, final_num_input_channels_model,
                                                     sequence_length)  # (N, C, L)
        else:
            X_test_flat_scaled = final_scaler.transform(X_test_flat_df)
            X_test_seq_final = X_test_flat_scaled.reshape(n_test, original_num_base_features,
                                                          sequence_length)  # (N, C, L)
    except ValueError as e:
        print(f"Ошибка подготовки Test данных TCN: {e}");
        traceback.print_exc();
        return None
    X_test_tensor_final = torch.tensor(X_test_seq_final, dtype=torch.float32)
    print(f"Финальные формы TCN: Train+Val={X_train_val_tensor.shape}, Test={X_test_tensor_final.shape}")

    # --- Создание и обучение финальной TCN модели ---
    try:  # Генерируем каналы для финальной модели
        fixed_trial_tcn_cv = optuna.trial.FixedTrial(best_params_tcn_cv)
        final_tcn_channels = generate_channels(fixed_trial_tcn_cv, best_params_tcn_cv['num_layers'],
                                               final_num_input_channels_model)
    except Exception as e:
        print(f"Ошибка ген. каналов фин. TCN: {e}. Дефолт.");
        final_tcn_channels = [max(16,
                                  final_num_input_channels_model)] * best_params_tcn_cv.get(
            'num_layers', 2)

    final_model_params = {
        'num_input_channels_model': final_num_input_channels_model, 'output_size': 1,
        'num_channels': final_tcn_channels, 'kernel_size': best_params_tcn_cv['kernel_size'],
        'dropout_rate': best_params_tcn_cv['dropout_rate'], 'activation_fn': best_params_tcn_cv['activation'],
        'dilation_base': best_params_tcn_cv['dilation_base']}
    final_train_params = {
        'optimizer_name': best_params_tcn_cv['optimizer_name'], 'learning_rate': best_params_tcn_cv['learning_rate'],
        'weight_decay': best_params_tcn_cv['weight_decay'], 'epochs': 300,  # Больше эпох
        'batch_size': best_params_tcn_cv['batch_size'], 'patience': 30,  # Больше терпения
        'scheduler_name': best_params_tcn_cv.get('scheduler'), 'final_training': True}

    # Обучаем на ВСЕХ Train+Val, валидация по Test для early stopping
    final_model_tcn, final_history, final_best_val_loss_on_test = train_single_fold(
        X_train_val_tensor, y_train_val_series, X_test_tensor_final, y_test,
        SequentialTCNModel, final_model_params, final_train_params, device)

    # 6. Оценка Финальной TCN Модели на Тесте
    print(f"\nФинальная оценка TCN модели ({pca_info}) для {current_period_label} на тесте:")
    test_mse, test_rmse, r2, y_pred = evaluate_model(final_model_tcn, X_test_tensor_final, y_test, device)
    # Расчет AIC/BIC
    num_parameters = sum(p.numel() for p in final_model_tcn.parameters() if p.requires_grad);
    n = len(y_test);
    aic, bic = np.nan, np.nan
    if not np.isnan(test_mse) and test_mse > 0 and n > 0:
        try:
            log_mse = np.log(
                test_mse);
            aic = n * log_mse + 2 * num_parameters;
            bic = n * log_mse + num_parameters * np.log(
                n) if n > 1 else aic
        except(RuntimeWarning, ValueError):
            pass
    else:
        print(f"Warn: Не рассчитать AIC/BIC (MSE={test_mse:.4f}, n={n}).")
    print(f"Тест MSE: {test_mse:.5f}, RMSE: {test_rmse:.5f}, R²: {r2:.5f}");
    print(f"AIC: {aic:.2f}, BIC: {bic:.2f}");
    print(f"Кол-во параметров TCN: {num_parameters}");
    print(f"Время обработки: {time.time() - start_time_period:.2f} сек.")

    # 7. Графики (RMSE финального обучения, Pred vs Real, Архитектура)
    # ... (код графиков такой же, как для TCN в предыдущем ответе) ...
    epochs_ran = len(final_history.get('val_loss', []));
    fig_height = 5;
    fig_width = 18
    plt.figure(figsize=(fig_width, fig_height))
    ax1 = plt.subplot(1, 3, 1)
    if epochs_ran > 0 and 'train_rmse' in final_history and 'val_rmse' in final_history:
        ax1.plot(range(epochs_ran), final_history['train_rmse'], label='Train RMSE (on Train+Val)');
        ax1.plot(range(epochs_ran), final_history['val_rmse'], label='Validation RMSE (on Test)');
        ax1.set_xlabel('Эпохи');
        ax1.set_ylabel('RMSE');
        ax1.set_title(f'TCN Финал. Обучение (RMSE) - до {current_period_label} {pca_info}');
        ax1.legend();
        ax1.grid(True)
        valid_tr = [r for r in final_history['train_rmse'] if pd.notna(r) and np.isfinite(r)];
        valid_v = [r for r in final_history['val_rmse'] if pd.notna(r) and np.isfinite(r)];
        if valid_tr and valid_v:
            min_r = min(min(valid_tr), min(valid_v));
            max_r = max(max(valid_tr),
                        np.percentile(valid_v, 98) if valid_v else max(
                            valid_tr));
            ax1.set_ylim(
                bottom=max(0, min_r * 0.9), top=max_r * 1.1 if max_r > 0 else 1)
        elif valid_tr:
            ax1.set_ylim(bottom=0, top=max(valid_tr) * 1.1 if max(valid_tr) > 0 else 1)
        else:
            ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, 'Нет данных\nдля графика RMSE', ha='center', va='center');
        ax1.set_title(
            f'TCN Финал. Обучение (RMSE) - до {current_period_label} {pca_info}')
    ax2 = plt.subplot(1, 3, 2)
    if not np.isnan(r2) and y_pred is not None and len(y_test) > 0 and len(y_pred) == len(y_test):
        valid_idx = ~np.isnan(y_pred) & ~np.isinf(y_pred);
        ax2.scatter(y_test.values[valid_idx], y_pred[valid_idx], alpha=0.6, label='Предсказания');
        valid_t = y_test.values[valid_idx]
        if len(valid_t) > 0: min_v = min(valid_t); max_v = max(valid_t); ax2.plot([min_v, max_v], [min_v, max_v], '--r',
                                                                                  linewidth=2, label='Идеал')
        ax2.set_xlabel("Реальные");
        ax2.set_ylabel("Предсказанные");
        ax2.set_title(f"TCN (до {current_period_label} {pca_info}) - Прогноз vs Реальность (R²: {r2:.3f})");
        ax2.legend();
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Не построить график\n(NaN в R2 или y_pred)', ha='center', va='center');
        ax2.set_title(
            f"TCN (до {current_period_label} {pca_info}) - Прогноз vs Реальность")
    ax3 = plt.subplot(1, 3, 3);
    ax3.axis('off');
    ax3.set_title(f"Архитектура TCN ({current_period_label})")
    architecture_viz_path = None
    try:  # Попытка torchviz
        import torchviz
        sample_input_shape_tcn = (
            1, final_num_input_channels_model, sequence_length)  # Используем размер после PCA/Scale
        x_viz = torch.randn(sample_input_shape_tcn, device=device)
        output_viz = final_model_tcn(x_viz)
        period_label_safe = current_period_label.replace('%', 'pct');
        viz_filename_base = os.path.join(models_save_dir, f"tcn_arch_period_{period_label_safe}_pca_{use_pca}")
        os.makedirs(models_save_dir, exist_ok=True);
        graph = torchviz.make_dot(output_viz, params=dict(final_model_tcn.named_parameters()), show_attrs=True,
                                  show_saved=True)
        graph.render(viz_filename_base, format='png', cleanup=True);
        architecture_viz_path = f"{viz_filename_base}.png";
        print(f"Граф TCN сохранен: {architecture_viz_path}")
        try:
            img = plt.imread(architecture_viz_path);
            ax3.imshow(img)
        except Exception as img_e:
            print(f"Ошибка отображения PNG: {img_e}");
            ax3.text(0.5, 0.5, "Граф сохранен,\nно не отобразить.",
                     ha='center', va='center', fontsize=9)
    except ImportError:
        print("torchviz не найден. Вывод torchinfo.")
    except Exception as viz_e:
        print(f"Ошибка viz (Graphviz?): {viz_e}")
    finally:
        if 'torchviz' not in sys.modules or architecture_viz_path is None:
            try:
                sample_input_shape_tcn = (1, final_num_input_channels_model, sequence_length)
                summary_str = summary(final_model_tcn, input_size=sample_input_shape_tcn, verbose=0, device=str(device),
                                      col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
                error_msg = f"Ошибка Graphviz:\n{viz_e}\n\n" if 'viz_e' in locals() and viz_e else ""
                ax3.text(0.01, 0.99, f"{error_msg}Сводка:\n{summary_str}", va='top', ha='left', wrap=True, fontsize=6,
                         family='monospace')
            except Exception as summary_e:
                print(f"Ошибка torchinfo: {summary_e}");
                ax3.text(0.5, 0.5, "Ошибка генерации\nсводки.", ha='center',
                         va='center', fontsize=9)
    plt.tight_layout(pad=1.5, w_pad=2.0);
    plt.show()

    # 8. Сохранение данных для инференса TCN
    # ... (код сохранения такой же, как в предыдущем TCN примере) ...
    print(f"Сохранение данных для инференса TCN периода {current_period_label}...")
    saved_inference_data_path = None
    try:
        os.makedirs(models_save_dir, exist_ok=True)
        inference_data_tcn = {
            'model_state_dict': final_model_tcn.state_dict(),
            'model_class_name': final_model_tcn.__class__.__name__,
            'model_init_params': final_model_params,  # Используем параметры финальной модели
            'feature_columns_ordered': feature_columns_ordered,
            'scaler': final_scaler, 'pca': final_pca_model,
            'metadata': {'base_feature_names': base_feature_names, 'sequence_length': sequence_length,
                         'pca_used': use_pca, 'original_num_base_features': original_num_base_features,
                         'input_reshape_order': 'NCL'}}
        period_label_safe = current_period_label.replace('%', 'pct')
        inference_filename = os.path.join(models_save_dir,
                                          f"inference_tcn_period_{period_label_safe}_pca_{use_pca}.pth")
        torch.save(inference_data_tcn, inference_filename)
        print(f"Данные для инференса TCN сохранены в: {inference_filename}")
        # saved_inference_data_path = inference_filename
    except Exception as e:
        print(f"Ошибка сохранения TCN данных {current_period_label}: {e}");
        traceback.print_exc();

    # 9. Формирование словаря results в нужном формате
    # Получаем RMSE на финальной обучающей выборке (Train+Val)
    train_val_rmse = np.nan
    if final_history and 'train_rmse' in final_history and final_history['train_rmse']:
        last_train_rmse = final_history['train_rmse'][-1]
        if pd.notna(last_train_rmse) and np.isfinite(last_train_rmse):
            train_val_rmse = last_train_rmse
        else:
            print(f"Warn: Последнее Train RMSE некорректно ({last_train_rmse}) для TCN {current_period_label}")
    results = {
        'Model': "TCN_CV",  # <<< Указываем TCN_CV
        'Period': current_period_label,
        'Best Parameters': best_params_tcn_cv,  # Параметры от Optuna CV
        'Train RMSE': train_val_rmse,  # RMSE на финальном обучающем наборе (Train+Val)
        'Test RMSE': test_rmse,  # RMSE на тестовом наборе
        'Test MSE': test_mse,  # MSE на тестовом наборе
        'R²': r2,  # R^2 на тестовом наборе
        'BIC': bic,  # BIC на тестовом наборе (приближенный)
        'AIC': aic  # AIC на тестовом наборе (приближенный)
    }
    print(f"Сформирован словарь результатов для TCN_CV периода {current_period_label}.")

    return results


# --- Основной цикл запуска для TCN с CV ---
if __name__ == "__main__":
    all_results_tcn_cv = []
    N_OPTUNA_TRIALS_TCN_CV = 1  # Уменьшим для теста
    USE_PCA_TCN_CV = True
    MODELS_SAVE_DIR_TCN_CV = "saved_tcn_models_cv"  # Отдельная папка

    print(
        f"\n=== Начало процесса оптимизации TCN с CV ({N_OPTUNA_TRIALS_TCN_CV} триалов/период, PCA: {USE_PCA_TCN_CV}) ===")
    print(f"Модели TCN CV будут сохранены в папку: {MODELS_SAVE_DIR_TCN_CV}")
    total_start_time_tcn_cv = time.time()

    for idx, _ in enumerate(periods):
        period_result_tcn_cv = process_period_tcn_cv(  # <<< Вызываем новую функцию
            idx, df_cleaned, base_feature_names, device,
            n_trials=N_OPTUNA_TRIALS_TCN_CV, use_pca=USE_PCA_TCN_CV, pca_variance=0.9,
            models_save_dir=MODELS_SAVE_DIR_TCN_CV, n_splits_optuna=5)  # <<< Указываем число фолдов
        if period_result_tcn_cv: all_results_tcn_cv.append(period_result_tcn_cv)

    print(f"\nПроцесс TCN CV завершен. Общее время: {(time.time() - total_start_time_tcn_cv) / 60:.2f} мин.")

    # --- Вывод итоговых результатов TCN CV ---
    if all_results_tcn_cv:
        results_df_tcn_cv = pd.DataFrame(all_results_tcn_cv)
        print("\n===== Итоговые результаты по периодам (TCN_CV модели) =====")
        cols_to_print = ['Model', 'Period', 'Best Parameters', 'Train RMSE', 'Test MSE', 'Test RMSE', 'R²', 'BIC',
                         'AIC']
        cols_to_print = [col for col in cols_to_print if col in results_df_tcn_cv.columns]
        print(results_df_tcn_cv[cols_to_print].round(4).to_string(index=False))

        print("\nЛучшие параметры TCN_CV:")
        for index, row in results_df_tcn_cv.iterrows():
            print(f"\n--- TCN_CV Период: {row['Period']} ---");
            best_p = row['Best Parameters']
            print("  Best Parameters:")
            for k, v in best_p.items():
                if isinstance(v, (list, dict)) and len(str(v)) > 50:
                    print(f"    {k}: [Too long]")
                else:
                    print(f"    {k}: {v}")
        try:
            fn = f"tcn_cv_optuna_summary_pca_{USE_PCA_TCN_CV}.xlsx";
            results_df_tcn_cv[cols_to_print].to_excel(fn, index=False, engine='openpyxl');
            print(f"\nСводка TCN_CV сохранена: {fn}")
        except Exception as e:
            print(f"\nОшибка сохранения сводки TCN_CV: {e}");
            traceback.print_exc()
    else:
        print("\nНе удалось получить результаты TCN_CV (Optuna).")
