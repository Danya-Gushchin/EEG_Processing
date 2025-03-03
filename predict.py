import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from network import DeepSeparator
from asammdf import MDF
import pandas as pd


model_name = 'DeepSeparator'


# test_input = np.load('data/test_input.npy')  
# test_output = np.load('data/test_output.npy') 
# print(np.shape(test_input))


# test_input = torch.from_numpy(test_input).float()
# test_output = torch.from_numpy(test_output).float()
# print(test_input.shape)

# test_dataset = Data.TensorDataset(test_input, test_output)
# test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Загружаем CSV
csv_path = "data/s00.csv"  # Укажи правильный путь к файлу
df = pd.read_csv(csv_path, header=None)
print(df.shape)

# Посмотрим на структуру данных
# print(df.head())

# Выбираем только числовые данные (без временных меток, если они есть)
eeg_data = df.select_dtypes(include=[np.number]).values  # (samples, num_channels)


# Преобразуем в Torch Tensor
test_input = torch.from_numpy(eeg_data).float()
# test_input = torch.unsqueeze(test_input, 0)

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


model.eval()


all_raw_eeg = []
all_denoised_eeg = []
# all_clean_eeg = []

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
# all_clean_eeg = np.concatenate(all_clean_eeg, axis=0)

# Визуализация
for i in range(len(all_raw_eeg)):
    plt.figure(figsize=(10, 6))
    l0, = plt.plot(all_raw_eeg[:,i], label='Raw EEG')
    l1, = plt.plot(all_denoised_eeg[:,i], label='Denoised EEG')
    # l2, = plt.plot(all_clean_eeg[i], label='Clean EEG')
    plt.legend(loc='upper right')
    plt.title(f'Sample {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.show()
