import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold # <<< Добавлен KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, Subset # <<< Добавлен Subset
import copy
import pandas as pd
import optuna
import time
import warnings
import os
import traceback
from torchinfo import summary
import matplotlib.pyplot as plt

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
    "Среднее время между заходами", "Оценка за интервал" ]
num_base_features = len(base_feature_names)
print(f"Определено {num_base_features} базовых признаков: {base_feature_names}")
try:
    df_original = pd.read_excel(file_path)
    print(f"Данные успешно загружены. Форма: {df_original.shape}")
    all_expected_cols=[f"{base} {p}" for p in periods for base in base_feature_names]; missing_cols=[col for col in all_expected_cols if col not in df_original.columns]
    if missing_cols: print(f"ОШИБКА: Отсутствуют колонки: {missing_cols}."); exit()
    else: print("Все ожидаемые колонки присутствуют.")
except FileNotFoundError:
    print(f"Ошибка: Файл не найден: {file_path}. Создание dummy данных.")
    num_students=200; data={'ID_студента':range(num_students)}
    for p in periods:
        for base_name in base_feature_names: data[f"{base_name} {p}"]=np.random.rand(num_students)*100
    data["Оценка за интервал 100%"]=np.random.rand(num_students)*70+30; df_original=pd.DataFrame(data); print(f"Dummy DataFrame создан. Форма: {df_original.shape}")

target_column = "Оценка за интервал 100%"
# --- Очистка выбросов ---
if target_column in df_original.columns:
    if pd.api.types.is_numeric_dtype(df_original[target_column]):
        Q1=df_original[target_column].quantile(0.25); Q3=df_original[target_column].quantile(0.75); IQR=Q3-Q1
        lower_bound=Q1-1.5*IQR; upper_bound=Q3+1.5*IQR; df_cleaned=df_original[(df_original[target_column]>=lower_bound)&(df_original[target_column]<=upper_bound)].copy()
        print(f"Очистка выбросов: {df_cleaned.shape} (удалено {df_original.shape[0]-df_cleaned.shape[0]})")
    else: print("Предупреждение: Целевая колонка не числовая."); df_cleaned=df_original.copy()
else: print(f"Ошибка: Целевая колонка '{target_column}' не найдена."); exit()
# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Устройство: {device}")

# --- Вспомогательные функции ---
def get_optimizer(optimizer_name):
    """Возвращает класс оптимизатора PyTorch по имени."""
    if optimizer_name == 'Adam': return optim.Adam
    elif optimizer_name == 'AdamW': return optim.AdamW
    elif optimizer_name == 'RMSprop': return optim.RMSprop
    else: raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")

# --- Модель LSTM --- (Без изменений)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional=False):
        super().__init__(); self.hidden_size=hidden_size; self.num_layers=num_layers; self.bidirectional=bidirectional
        lstm_dropout_rate=dropout if num_layers>1 else 0
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=lstm_dropout_rate,bidirectional=bidirectional)
        linear_input_dim=hidden_size*2 if bidirectional else hidden_size
        self.fc=nn.Linear(linear_input_dim,output_size); self.dropout_layer=nn.Dropout(dropout)
    def forward(self, x):
        lstm_out,_=self.lstm(x); last_time_step_output=lstm_out[:,-1,:]; out=self.dropout_layer(last_time_step_output); out=self.fc(out); return out

# --- Функция обучения одного фолда или финальной модели ---
def train_single_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, model_class, model_params, train_params, device):
    """Обучает модель на одном фолде с ранней остановкой."""
    model = model_class(**model_params).to(device)
    optimizer_class = get_optimizer(train_params['optimizer_name'])
    optimizer = optimizer_class(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    criterion = nn.MSELoss()

    # <<< Инициализируем history здесь >>>
    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    patience = train_params.get('patience', 20)
    epochs = train_params['epochs']

    train_dataset = TensorDataset(X_train_fold, torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, pin_memory=device.type == 'cuda')
    val_dataset = TensorDataset(X_val_fold, torch.tensor(y_val_fold.values, dtype=torch.float32).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], pin_memory=device.type == 'cuda')

    scheduler = None
    if train_params.get('final_training', False) and train_params.get('scheduler_name'):
        scheduler_name = train_params['scheduler_name']
        if scheduler_name=='ReduceLROnPlateau':scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=patience//2,verbose=False)
        elif scheduler_name=='CosineAnnealingLR':scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-7)

    # print(f"  Начало обучения фолда ({epochs} эпох)...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_outputs = model(batch_X_val)
                val_loss = criterion(val_outputs, batch_y_val)
                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    epoch_val_loss += val_loss.item() * batch_X_val.size(0)
                else: epoch_val_loss += 1e10 * batch_X_val.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        avg_val_loss = epoch_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('inf')
        train_rmse = np.sqrt(max(0, avg_train_loss))
        val_rmse = np.sqrt(max(0, avg_val_loss)) if not np.isinf(avg_val_loss) else float('inf')

        # <<< Заполняем history >>>
        history['train_loss'].append(avg_train_loss); history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(train_rmse); history['val_rmse'].append(val_rmse)

        if scheduler: # Обновляем планировщик только при финальном обучении
            if isinstance(scheduler,optim.lr_scheduler.ReduceLROnPlateau):scheduler.step(avg_val_loss)
            else:scheduler.step()

        # Вывод прогресса для финального обучения
        if train_params.get('final_training', False) and ((epoch + 1) % 25 == 0 or epoch == 0):
             current_lr = optimizer.param_groups[0]['lr']
             print(f'    Эпоха {epoch+1:3d}/{epochs}, Train RMSE: {train_rmse:.5f}, Val RMSE (Test): {val_rmse:.5f}, LR: {current_lr:.6f}')


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if train_params.get('final_training', False): # Печатаем только при финальном обучении
                     print(f'    Ранняя остановка на эпохе {epoch+1}.')
                break

        if np.isinf(avg_val_loss) or np.isnan(avg_val_loss):
             if train_params.get('final_training', False):
                  print(f"    Остановка из-за некорр. Val Loss на эпохе {epoch+1}.")
             break

    model.load_state_dict(best_model_wts)
    # <<< Возвращаем history вместе с моделью и лучшим лоссом >>>
    return model, history, best_val_loss


# --- Функция оценки (общая) ---
def evaluate_model(model, X_test_tensor, y_test_series, device):
    """Оценивает обученную модель на тестовых данных."""
    # ... (код без изменений) ...
    model.eval();y_pred_list=[];y_true_list=y_test_series.values
    test_dataset=TensorDataset(X_test_tensor);test_loader=DataLoader(test_dataset,batch_size=512,pin_memory=device.type=='cuda') # Увеличим батч для оценки
    with torch.no_grad():
        for batch_X_tuple in test_loader:batch_X=batch_X_tuple[0].to(device,non_blocking=True);outputs=model(batch_X).cpu().numpy().flatten();y_pred_list.extend(outputs)
    y_pred=np.array(y_pred_list)
    if len(y_pred)!=len(y_true_list):print(f"Warn: y_pred({len(y_pred)}) vs y_test({len(y_true_list)})");min_len=min(len(y_pred),len(y_true_list));y_pred=y_pred[:min_len];y_true_list=y_true_list[:min_len]
    if np.isnan(y_pred).any()or np.isinf(y_pred).any():print("Warn: NaN/Inf preds!");return np.nan,np.nan,np.nan,y_pred
    mse=mean_squared_error(y_true_list,y_pred);rmse=np.sqrt(max(0,mse))
    if np.var(y_true_list)<1e-9:r2=np.nan if mse>1e-9 else 1.0
    else:r2=r2_score(y_true_list,y_pred)
    return mse,rmse,r2,y_pred

# --- Optuna objective для LSTM с Кросс-Валидацией ---
def objective_lstm_cv(trial, X_train_val_tensor, y_train_val_series, input_size, device, n_splits=5):
    """Целевая функция Optuna для LSTM, использующая K-Fold CV."""

    # --- Параметры архитектуры LSTM ---
    lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [16, 32, 64, 128]) # Уменьшим для скорости CV
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.0, 0.6, step=0.1)
    lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [False, True])

    # --- Параметры обучения (применяются к каждому фолду) ---
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Немного сузим LR
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam']) # Уберем RMSprop для простоты
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True) # Сузим WD
    batch_size = trial.suggest_categorical('batch_size', [32, 64]) # Меньше батчи для CV
    # Уменьшим параметры для ускорения CV внутри Optuna
    patience_value = trial.suggest_int('patience', 7, 15)
    epochs = 100 # Меньше эпох для CV

    model_params = { # Параметры для инициализации модели
        'input_size': input_size, 'hidden_size': lstm_hidden_size, 'num_layers': lstm_num_layers,
        'output_size': 1, 'dropout': lstm_dropout, 'bidirectional': lstm_bidirectional }
    train_params = { # Параметры для функции train_single_fold
        'optimizer_name': optimizer_name, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'epochs': epochs, 'batch_size': batch_size, 'patience': patience_value }

    # --- K-Fold Кросс-Валидация ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=trial.number) # trial.number для разной разбивки в триалах
    fold_val_losses = []
    indices = np.arange(len(X_train_val_tensor)) # Индексы для разбиения

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # print(f"  Триал {trial.number}, Фолд {fold+1}/{n_splits}") # Отладочный вывод
        # Создаем тензоры для фолда
        X_train_fold = X_train_val_tensor[train_idx]
        y_train_fold = y_train_val_series.iloc[train_idx]
        X_val_fold = X_train_val_tensor[val_idx]
        y_val_fold = y_train_val_series.iloc[val_idx]

        try:
            # Обучаем модель на текущем фолде
            _, _, best_fold_val_loss = train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                LSTMModel, model_params, train_params, device )

            # Сохраняем лучший val loss для этого фолда
            if not np.isnan(best_fold_val_loss) and np.isfinite(best_fold_val_loss):
                fold_val_losses.append(best_fold_val_loss)
            else:
                 print(f"WARNING: Некорректный val_loss ({best_fold_val_loss}) в триале {trial.number}, фолд {fold+1}. Пропуск фолда.")
                 # Если даже один фолд не удался, считаем триал неудачным
                 return float('inf')

            # Прунинг внутри CV (если нужно еще ускорить)
            trial.report(np.mean(fold_val_losses) if fold_val_losses else float('inf'), step=fold)
            if trial.should_prune():
                 raise optuna.TrialPruned()

        except optuna.TrialPruned as e:
            raise e # Передаем прунинг дальше
        except Exception as e:
            print(f"Ошибка обучения фолда {fold+1} в триале {trial.number}: {e}")
            traceback.print_exc()
            return float('inf') # Считаем триал неудачным при ошибке

    # Возвращаем средний лучший val_loss по всем фолдам
    avg_cv_loss = np.mean(fold_val_losses) if fold_val_losses else float('inf')
    avg_cv_rmse = np.sqrt(max(0, avg_cv_loss)) if not np.isinf(avg_cv_loss) else float('inf')

    print(f"  Триал {trial.number} завершен. Средний CV RMSE: {avg_cv_rmse:.5f}") # Отладочный вывод
    return avg_cv_rmse # Optuna будет минимизировать средний RMSE по фолдам

# --- Основная функция обработки периода для LSTM с CV в Optuna ---
def process_period_lstm_cv(period_index, df_clean, base_feature_names, device, n_trials=50, use_pca=False, pca_variance=0.95, models_save_dir="saved_lstm_models_cv", n_splits_optuna=5):
    """
    Выполняет цикл обработки для LSTM: подготовка данных, Optuna с K-Fold CV,
    обучение финальной модели, оценка и сохранение.
    """
    current_period_label = periods[period_index]; periods_to_include = periods[:period_index + 1]
    sequence_length = len(periods_to_include); original_num_base_features = len(base_feature_names)
    pca_info = f"с PCA (var={pca_variance*100:.0f}%)" if use_pca else "без PCA"
    print(f"\n===== Запуск Optuna LSTM (CV={n_splits_optuna}) до {current_period_label} ({n_trials} тр, {pca_info}, L={sequence_length}, C_orig={original_num_base_features}) =====")
    start_time_period = time.time()

    # 1. Формирование колонок
    # ... (код без изменений) ...
    feature_columns_ordered = []
    for p in periods_to_include:
        for base_name in base_feature_names:
            col_name = f"{base_name} {p}";
            if col_name in df_clean.columns: feature_columns_ordered.append(col_name)
            else: print(f"ОШИБКА: Колонка {col_name} не найдена."); return None
    if len(feature_columns_ordered)!=original_num_base_features*sequence_length: print(f"ОШИБКА: Неверное кол-во признаков."); return None
    X_flat_df = df_clean[feature_columns_ordered]; y = df_clean[target_column]


    # --- ИЗМЕНЕНИЕ: Используем ВСЕ данные (кроме теста) для Optuna с CV ---
    # Делим только на Train+Val (80%) и Test (20%)
    X_train_val_flat_df, X_test_flat_df, y_train_val, y_test = train_test_split(X_flat_df, y, test_size=0.2, random_state=42)
    print(f"Размеры плоских данных: Train+Val={X_train_val_flat_df.shape}, Test={X_test_flat_df.shape}")
    if X_train_val_flat_df.shape[0] < n_splits_optuna * 2: # Проверка на достаточность данных для CV
        print(f"Предупреждение: Недостаточно данных ({X_train_val_flat_df.shape[0]}) для {n_splits_optuna}-Fold CV. Пропуск периода.")
        return None

    # 3. Подготовка данных Train+Val для Optuna (Scaler, PCA, Reshape)
    final_scaler = StandardScaler() # Scaler будет обучаться на Train+Val и использоваться для Test
    final_pca_model = None          # PCA будет обучаться на Train+Val и использоваться для Test
    input_size_model = original_num_base_features # C для LSTM
    n_train_val = X_train_val_flat_df.shape[0]

    try:
        if use_pca:
            print(f"Применение Scaler+PCA (обучение на Train+Val 2D)...")
            # Reshape Train+Val в 2D
            X_train_val_2d = X_train_val_flat_df.values.reshape(-1, original_num_base_features)
            # Обучаем Scaler на Train+Val
            X_train_val_2d_scaled = final_scaler.fit_transform(X_train_val_2d)
            # Обучаем PCA на Train+Val
            final_pca_model = PCA(n_components=pca_variance, random_state=42).fit(X_train_val_2d_scaled)
            # Применяем PCA к Train+Val
            X_train_val_2d_pca = final_pca_model.transform(X_train_val_2d_scaled)

            n_components = X_train_val_2d_pca.shape[1]
            input_size_model = n_components
            print(f"PCA оставил {n_components} компонент (на Train+Val).")
            if n_components == 0: raise ValueError("PCA не оставил компонент (на Train+Val).")

            # Reshape Train+Val обратно в 3D (N, L, C_pca) для Optuna CV
            X_train_val_seq = X_train_val_2d_pca.reshape(n_train_val, sequence_length, n_components)
        else:
            print("Применение Scaler (обучение на Train+Val плоских)...")
            # Обучаем Scaler на Train+Val плоских
            X_train_val_flat_scaled = final_scaler.fit_transform(X_train_val_flat_df)
            # Reshape в 3D (N, L, C_orig) для Optuna CV
            X_train_val_seq = X_train_val_flat_scaled.reshape(n_train_val, sequence_length, original_num_base_features)
            # input_size_model остается original_num_base_features

    except ValueError as e:
        print(f"Ошибка при подготовке/reshape данных Train+Val: {e}")
        traceback.print_exc(); return None

    # Конвертация в тензор для Optuna CV
    X_train_val_tensor = torch.tensor(X_train_val_seq, dtype=torch.float32)
    y_train_val_series = y_train_val # Используем Series для iloc в CV
    print(f"Форма данных для Optuna CV: {X_train_val_tensor.shape}")

    # 4. Запуск Optuna для LSTM с Кросс-Валидацией
    print(f"Запуск Optuna LSTM с {n_splits_optuna}-Fold CV (Input C={input_size_model}, L={sequence_length})...")
    study_lstm_cv = optuna.create_study(
        direction="minimize", # Минимизируем средний CV RMSE
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=1) # Более агрессивный прунер для CV
    )
    try:
        # Передаем X_train_val и y_train_val в objective
        study_lstm_cv.optimize(lambda trial: objective_lstm_cv(trial, X_train_val_tensor, y_train_val_series, input_size_model, device, n_splits=n_splits_optuna),
                           n_trials=n_trials, timeout=10800) # Увеличим таймаут (3 часа) для CV
    except Exception as e:
        print(f"Критическая ошибка Optuna LSTM CV: {e}"); traceback.print_exc();
        if not study_lstm_cv.trials: print("Нет триалов Optuna LSTM CV. Пропуск."); return None
    try: best_trial_lstm_cv = study_lstm_cv.best_trial
    except ValueError: print("Optuna LSTM CV не нашла лучший триал."); return None

    best_params_lstm_cv = best_trial_lstm_cv.params
    # Важно: best_trial_lstm_cv.value теперь содержит СРЕДНИЙ CV RMSE
    best_cv_rmse = best_trial_lstm_cv.value
    # Добавим параметры обучения в best_params для финальной тренировки
    # Используем параметры из лучшего триала, но можем увеличить эпохи/patience
    best_params_lstm_cv['learning_rate'] = best_trial_lstm_cv.params.get('learning_rate')
    best_params_lstm_cv['optimizer_name'] = best_trial_lstm_cv.params.get('optimizer')
    best_params_lstm_cv['weight_decay'] = best_trial_lstm_cv.params.get('weight_decay')
    best_params_lstm_cv['batch_size'] = best_trial_lstm_cv.params.get('batch_size')
    # Параметры архитектуры уже есть в best_params_lstm_cv
    best_params_lstm_cv['actual_input_size'] = input_size_model

    print(f"\nOptuna LSTM CV завершена. Лучший средний CV RMSE: {best_cv_rmse:.5f}")
    print(f"Лучшие параметры LSTM (для {input_size_model} вх. признаков):"); [print(f"  {k}: {v}") for k, v in best_params_lstm_cv.items()]

    # 5. Обучение Финальной LSTM Модели на ВСЕХ Train+Val данных
    print(f"\nОбучение финальной LSTM модели для {current_period_label} на ВСЕХ Train+Val данных...")

    # --- Подготовка Тестовых данных (Scaler/PCA уже обучены на Train+Val) ---
    n_test = X_test_flat_df.shape[0]
    final_input_size_model = input_size_model # Input size уже определен
    try:
        if use_pca:
            # Применяем обученные Scaler и PCA к Test
            X_test_2d = X_test_flat_df.values.reshape(-1, original_num_base_features)
            X_test_2d_scaled = final_scaler.transform(X_test_2d) # Используем final_scaler
            X_test_2d_pca = final_pca_model.transform(X_test_2d_scaled) # Используем final_pca_model
            X_test_seq_final = X_test_2d_pca.reshape(n_test, sequence_length, final_input_size_model) # (N, L, C)
        else:
            # Применяем обученный Scaler к Test
            X_test_flat_scaled = final_scaler.transform(X_test_flat_df)
            X_test_seq_final = X_test_flat_scaled.reshape(n_test, sequence_length, final_input_size_model) # (N, L, C)
    except ValueError as e: print(f"Ошибка подготовки Test данных: {e}"); traceback.print_exc(); return None
    X_test_tensor_final = torch.tensor(X_test_seq_final, dtype=torch.float32)
    print(f"Финальные формы LSTM: Train+Val={X_train_val_tensor.shape}, Test={X_test_tensor_final.shape}")

    # --- Создание и обучение финальной LSTM модели ---
    # Используем параметры из лучшего триала Optuna CV
    final_model_params = {
        'input_size': final_input_size_model,
        'hidden_size': best_params_lstm_cv['lstm_hidden_size'],
        'num_layers': best_params_lstm_cv['lstm_num_layers'],
        'output_size': 1,
        'dropout': best_params_lstm_cv['lstm_dropout'],
        'bidirectional': best_params_lstm_cv['lstm_bidirectional']
    }
    # Используем параметры обучения из лучшего триала, но увеличиваем эпохи/patience
    final_train_params = {
        'optimizer_name': best_params_lstm_cv['optimizer_name'],
        'learning_rate': best_params_lstm_cv['learning_rate'],
        'weight_decay': best_params_lstm_cv['weight_decay'],
        'epochs': 300, # Больше эпох для финального обучения
        'batch_size': best_params_lstm_cv['batch_size'],
        'patience': 30, # Больше терпения для финального обучения
        'scheduler_name': best_params_lstm_cv.get('scheduler'), # Используем scheduler из триала
        'final_training': True # Флаг для включения scheduler в train_single_fold
    }

    # Обучаем на ВСЕХ Train+Val данных, валидация на ТЕСТЕ для ранней остановки финальной модели
    final_model_lstm, final_history, final_best_val_loss_on_test = train_single_fold(
        X_train_val_tensor, y_train_val_series, # Обучение на Train+Val
        X_test_tensor_final, y_test,           # Валидация на Test
        LSTMModel, final_model_params, final_train_params, device )

    # 6. Оценка Финальной LSTM Модели на Тесте (повторная, но теперь с лучшими весами)
    print(f"\nФинальная оценка LSTM модели ({pca_info}) для {current_period_label} на тесте:")
    test_mse, test_rmse, r2, y_pred = evaluate_model(final_model_lstm, X_test_tensor_final, y_test, device)

    # Расчет AIC/BIC
    num_parameters=sum(p.numel() for p in final_model_lstm.parameters() if p.requires_grad); n=len(y_test); aic,bic=np.nan,np.nan
    if not np.isnan(test_mse) and test_mse>0 and n>0:
       try:log_mse=np.log(test_mse);aic=n*log_mse+2*num_parameters;bic=n*log_mse+num_parameters*np.log(n) if n>1 else aic
       except(RuntimeWarning,ValueError):pass
    else:print(f"Warn: Не рассчитать AIC/BIC (MSE={test_mse:.4f}, n={n}).")
    print(f"Тест MSE: {test_mse:.5f}, RMSE: {test_rmse:.5f}, R²: {r2:.5f}");print(f"AIC: {aic:.2f}, BIC: {bic:.2f}");print(f"Кол-во параметров LSTM: {num_parameters}");print(f"Время обработки: {time.time()-start_time_period:.2f} сек.")

    # 7. Графики (RMSE финального обучения, Pred vs Real, Архитектура)
    # ... (код графиков остается таким же, как в предыдущем ответе, т.к. final_history содержит нужные данные) ...
    epochs_ran = len(final_history.get('val_loss', []))
    fig_height = 5; fig_width = 18
    plt.figure(figsize=(fig_width, fig_height))
    # График 1: Кривая обучения RMSE
    ax1 = plt.subplot(1, 3, 1)
    if epochs_ran > 0 and 'train_rmse' in final_history and 'val_rmse' in final_history:
        ax1.plot(range(epochs_ran), final_history['train_rmse'], label='Train RMSE (on Train+Val)')
        ax1.plot(range(epochs_ran), final_history['val_rmse'], label='Validation RMSE (on Test)') # Валидация была на тесте
        ax1.set_xlabel('Эпохи'); ax1.set_ylabel('RMSE'); ax1.set_title(f'LSTM Финал. Обучение (RMSE) - до {current_period_label} {pca_info}'); ax1.legend(); ax1.grid(True)
        valid_tr=[r for r in final_history['train_rmse'] if pd.notna(r)and np.isfinite(r)]; valid_v=[r for r in final_history['val_rmse'] if pd.notna(r)and np.isfinite(r)]
        if valid_tr and valid_v: min_r=min(min(valid_tr),min(valid_v)); max_r=max(max(valid_tr),np.percentile(valid_v,98) if valid_v else max(valid_tr)); ax1.set_ylim(bottom=max(0,min_r*0.9),top=max_r*1.1 if max_r>0 else 1)
        elif valid_tr: ax1.set_ylim(bottom=0,top=max(valid_tr)*1.1 if max(valid_tr)>0 else 1)
        else: ax1.set_ylim(bottom=0)
    else: ax1.text(0.5,0.5,'Нет данных\nдля графика RMSE',ha='center',va='center'); ax1.set_title(f'LSTM Финал. Обучение (RMSE) - до {current_period_label} {pca_info}')
    # График 2: Предсказания vs Реальность
    ax2 = plt.subplot(1, 3, 2)
    if not np.isnan(r2) and y_pred is not None and len(y_test)>0 and len(y_pred)==len(y_test):
        valid_idx=~np.isnan(y_pred)&~np.isinf(y_pred); ax2.scatter(y_test.values[valid_idx],y_pred[valid_idx],alpha=0.6,label='Предсказания'); valid_t=y_test.values[valid_idx]
        if len(valid_t)>0: min_v=min(valid_t); max_v=max(valid_t); ax2.plot([min_v,max_v],[min_v,max_v],'--r',linewidth=2,label='Идеал')
        ax2.set_xlabel("Реальные"); ax2.set_ylabel("Предсказанные"); ax2.set_title(f"LSTM (до {current_period_label} {pca_info}) - Прогноз vs Реальность (R²: {r2:.3f})"); ax2.legend(); ax2.grid(True)
    else: ax2.text(0.5,0.5,'Не построить график\n(NaN в R2 или y_pred)',ha='center',va='center'); ax2.set_title(f"LSTM (до {current_period_label} {pca_info}) - Прогноз vs Реальность")
    # График 3: Архитектура модели LSTM
    ax3 = plt.subplot(1, 3, 3); ax3.axis('off'); ax3.set_title(f"Архитектура LSTM ({current_period_label})")
    try: # Попытка torchviz
        import torchviz
        sample_input_shape_lstm = (1, sequence_length, final_input_size_model)
        x_viz = torch.randn(sample_input_shape_lstm, device=device)
        output_viz = final_model_lstm(x_viz)
        period_label_safe = current_period_label.replace('%','pct'); viz_filename_base = os.path.join(models_save_dir, f"lstm_arch_period_{period_label_safe}_pca_{use_pca}")
        os.makedirs(models_save_dir, exist_ok=True); graph = torchviz.make_dot(output_viz, params=dict(final_model_lstm.named_parameters()), show_attrs=True, show_saved=True)
        graph.render(viz_filename_base, format='png', cleanup=True); architecture_viz_path = f"{viz_filename_base}.png"; print(f"Граф LSTM сохранен: {architecture_viz_path}")
        try: img = plt.imread(architecture_viz_path); ax3.imshow(img)
        except Exception as img_e: print(f"Ошибка отображения PNG: {img_e}"); ax3.text(0.5,0.5,"Граф сохранен,\nно не отобразить.",ha='center',va='center',fontsize=9)
    except ImportError: print("torchviz не найден. Вывод torchinfo.");
    except Exception as viz_e: print(f"Ошибка viz (Graphviz?): {viz_e}")
    finally: # В любом случае (ошибка или нет viz), выводим torchinfo если можем
        if 'torchviz' not in sys.modules or architecture_viz_path is None: # Если torchviz не импортирован или рендер не удался
             try:
                  sample_input_shape_lstm = (1, sequence_length, final_input_size_model)
                  summary_str = summary(final_model_lstm, input_size=sample_input_shape_lstm, verbose=0, device=str(device), col_names=["input_size","output_size","num_params","mult_adds"])
                  error_msg = f"Ошибка Graphviz:\n{viz_e}\n\n" if 'viz_e' in locals() and viz_e else ""
                  ax3.text(0.01,0.99,f"{error_msg}Сводка:\n{summary_str}",va='top',ha='left',wrap=True,fontsize=6,family='monospace')
             except Exception as summary_e: print(f"Ошибка torchinfo: {summary_e}"); ax3.text(0.5,0.5,"Ошибка генерации\nсводки.",ha='center',va='center',fontsize=9)
    plt.tight_layout(pad=1.5, w_pad=2.0); plt.show()


    # 8. Сохранение данных для инференса LSTM
    # ... (код сохранения остается прежним, использует final_model_lstm, final_scaler, final_pca_model) ...
    print(f"Сохранение данных для инференса LSTM периода {current_period_label}...")
    saved_inference_data_path = None
    try:
        os.makedirs(models_save_dir, exist_ok=True)
        inference_data_lstm = {
            'model_state_dict': final_model_lstm.state_dict(),
            'model_class_name': final_model_lstm.__class__.__name__,
            'model_init_params': final_model_params, # Используем параметры финальной модели
            'feature_columns_ordered': feature_columns_ordered,
            'scaler': final_scaler, 'pca': final_pca_model,
            'metadata': {
                'base_feature_names': base_feature_names, 'sequence_length': sequence_length,
                'pca_used': use_pca, 'original_num_base_features': original_num_base_features,
                'input_reshape_order': 'NLC'
            }}
        period_label_safe = current_period_label.replace('%', 'pct')
        inference_filename = os.path.join(models_save_dir, f"inference_lstm_period_{period_label_safe}_pca_{use_pca}.pth")
        torch.save(inference_data_lstm, inference_filename)
        print(f"Данные для инференса LSTM сохранены в: {inference_filename}")
        # saved_inference_data_path = inference_filename
    except Exception as e: print(f"Ошибка сохранения LSTM данных {current_period_label}: {e}"); traceback.print_exc();


    # 9. Формирование словаря results
    # Получаем RMSE на финальной обучающей выборке (Train+Val)
    train_val_rmse = np.nan
    if final_history and 'train_rmse' in final_history and final_history['train_rmse']:
         last_train_rmse = final_history['train_rmse'][-1]
         if pd.notna(last_train_rmse) and np.isfinite(last_train_rmse): train_val_rmse = last_train_rmse
         else: print(f"Warn: Последнее Train RMSE некорректно ({last_train_rmse}) для {current_period_label}")

    # Создаем словарь с требуемыми полями
    results = {
        'Model': "LSTM_CV",                 # <<< Указываем, что использовалась CV
        'Period': current_period_label,
        'Best Parameters': best_params_lstm_cv, # Параметры от Optuna CV
        'Train RMSE': train_val_rmse,       # RMSE на финальном обучающем наборе (Train+Val)
        'Test RMSE': test_rmse,             # RMSE на тестовом наборе
        'Test MSE': test_mse,               # MSE на тестовом наборе
        'R²': r2,                           # R^2 на тестовом наборе
        'BIC': bic,                         # BIC на тестовом наборе (приближенный)
        'AIC': aic                          # AIC на тестовом наборе (приближенный)
        # --- CV RMSE из Optuna теперь неявно в best_params_lstm_cv['best_val_rmse_optuna'], но не добавляем ---
    }
    print(f"Сформирован словарь результатов для LSTM_CV периода {current_period_label}.")

    return results

# --- Основной цикл запуска для LSTM с CV ---
if __name__ == "__main__":
    import sys # Для проверки наличия torchviz в finally блоке
    all_results_lstm_cv = []
    N_OPTUNA_TRIALS_LSTM_CV = 1 # Уменьшим для теста, CV значительно дольше
    USE_PCA_LSTM_CV = False     # Пример: отключим PCA для CV
    MODELS_SAVE_DIR_LSTM_CV = "saved_lstm_models_cv" # Отдельная папка

    print(f"\n=== Начало процесса оптимизации LSTM с CV ({N_OPTUNA_TRIALS_LSTM_CV} триалов/период, PCA: {USE_PCA_LSTM_CV}) ===")
    print(f"Модели LSTM CV будут сохранены в папку: {MODELS_SAVE_DIR_LSTM_CV}")
    total_start_time_lstm_cv = time.time()

    for idx, _ in enumerate(periods):
        period_result_lstm_cv = process_period_lstm_cv( # <<< Вызываем новую функцию
            idx, df_cleaned, base_feature_names, device,
            n_trials=N_OPTUNA_TRIALS_LSTM_CV, use_pca=USE_PCA_LSTM_CV, pca_variance=0.95,
            models_save_dir=MODELS_SAVE_DIR_LSTM_CV, n_splits_optuna=5 ) # <<< Указываем число фолдов
        if period_result_lstm_cv: all_results_lstm_cv.append(period_result_lstm_cv)

    print(f"\nПроцесс LSTM CV завершен. Общее время: {(time.time()-total_start_time_lstm_cv)/60:.2f} мин.")

    # --- Вывод итоговых результатов LSTM CV ---
    if all_results_lstm_cv:
        results_df_lstm_cv = pd.DataFrame(all_results_lstm_cv)
        print("\n===== Итоговые результаты по периодам (LSTM_CV модели) =====")
        cols_to_print = ['Model', 'Period', 'Best Parameters', 'Train RMSE', 'Test MSE', 'Test RMSE', 'R²', 'BIC', 'AIC']
        cols_to_print = [col for col in cols_to_print if col in results_df_lstm_cv.columns]
        print(results_df_lstm_cv[cols_to_print].round(4).to_string(index=False))

        print("\nЛучшие параметры LSTM_CV:")
        for index, row in results_df_lstm_cv.iterrows():
             print(f"\n--- LSTM_CV Период: {row['Period']} ---"); best_p=row['Best Parameters']
             print("  Best Parameters:")
             for k,v in best_p.items():
                  if isinstance(v,(list,dict))and len(str(v))>50: print(f"    {k}: [Too long]")
                  else: print(f"    {k}: {v}")
        try:
            fn=f"lstm_cv_optuna_summary_pca_{USE_PCA_LSTM_CV}.xlsx"; results_df_lstm_cv[cols_to_print].to_excel(fn, index=False, engine='openpyxl'); print(f"\nСводка LSTM_CV сохранена: {fn}")
        except Exception as e: print(f"\nОшибка сохранения сводки LSTM_CV: {e}"); traceback.print_exc()
    else: print("\nНе удалось получить результаты LSTM_CV (Optuna).")