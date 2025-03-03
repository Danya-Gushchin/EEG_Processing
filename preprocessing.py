import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from network import DeepSeparator
# from asammdf import MDF
import pandas as pd

def denoised_signal(csv_path):
    model_name = 'DeepSeparator'

    # Загружаем CSV
    df = pd.read_csv(csv_path, delimiter=';', skiprows=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df.head())

    if df.empty:
        raise ValueError("CSV файл пуст или не содержит числовых данных.")

    # Выбираем только числовые данные (без временных меток, если они есть)
    eeg_data = df.values  # (samples, num_channels)
    # print(eeg_data)

    if eeg_data.shape[1] == 0:
        raise ValueError("Нет числовых данных для обработки.")

    # Преобразуем в Torch Tensor
    test_input = torch.from_numpy(eeg_data).float()

    # Создаём DataLoader
    test_dataset = Data.TensorDataset(test_input)
    test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

    print("Загружено данных:", test_input.shape)

    print("torch.cuda.is_available() =", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DeepSeparator().to(device)
    checkpoint_path = os.path.join('checkpoint', f'{model_name}.pkl')
    if os.path.exists(checkpoint_path):
        print('Loading model from checkpoint...')
        model.load_state_dict(torch.load(checkpoint_path))

    model.eval()  # predict mode

    all_raw_eeg = []
    all_denoised_eeg = []

    # Основной цикл обработки
    with torch.no_grad():
        for batch_input in test_loader:  # Убираем batch_output
            batch_input = batch_input[0].to(device)  # [0], потому что DataLoader возвращает tuple
            
            # Прогон данных через модель
            extracted_signal = model(batch_input, 0)  # 0 для денойза
            
            # Сохранение результатов
            all_raw_eeg.append(batch_input.cpu().numpy())
            all_denoised_eeg.append(extracted_signal.cpu().numpy())

    all_raw_eeg = np.concatenate(all_raw_eeg, axis=0)
    all_denoised_eeg = np.concatenate(all_denoised_eeg, axis=0)

    return all_raw_eeg, all_denoised_eeg


# # Визуализация
# for i in range(len(all_raw_eeg)):
#     plt.figure(figsize=(10, 6))
#     l0, = plt.plot(all_raw_eeg[:,i], label='Raw EEG')
#     l1, = plt.plot(all_denoised_eeg[:,i], label='Denoised EEG')
#     plt.legend(loc='upper right')
#     plt.title(f'Sample {i + 1}')
#     plt.xlabel('Time')
#     plt.ylabel('Signal Amplitude')
#     plt.grid(True)
#     plt.show()
