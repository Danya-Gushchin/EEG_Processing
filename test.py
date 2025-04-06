import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

# 1. Загрузка данных
def load_eeg_data(file_path, sfreq=256):
    """
    Улучшенная загрузка ЭЭГ данных из CSV с обработкой ошибок формата
    
    Parameters:
    -----------
    file_path : str
        Путь к CSV файлу с данными
    sfreq : float
        Частота дискретизации (по умолчанию 256 Гц)
    """
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', decimal='.', dtype=str, on_bad_lines='skip')

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')

        if df.empty:
            raise ValueError("CSV файл не содержит числовых данных")

        data = df.values.T

        # Удаление строк, полностью состоящих из нулей
        non_zero_mask = ~(np.all(data == 0, axis=1))
        data = data[non_zero_mask]
        ch_names = [f"EEG_{i+1}" for i in range(data.shape[0])]

        if np.isnan(data).any():
            print("Предупреждение: данные содержат NaN. Заменяем на 0.")
            data = np.nan_to_num(data)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        print(f"Загружено {data.shape[0]} каналов, {data.shape[1]} отсчётов на канал.")
        print(raw)
        return raw

    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        raise
    
# 2. Предобработка данных
def preprocess_eeg(raw, l_freq=1, h_freq=40, notch_freq=50):
    """
    Основные этапы предобработки ЭЭГ:
    - Фильтрация
    - Удаление артефактов
    - Режекторный фильтр
    """
    # Копируем данные, чтобы не изменять оригинал
    raw_filtered = raw.copy()
    
    # Применяем bandpass фильтр (1-40 Гц по умолчанию)
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='iir')
    
    # Применяем notch фильтр для удаления сетевых помех (50 Гц в России/Европе)
    raw_filtered.notch_filter(freqs=notch_freq)
    
    # Автоматическое удаление артефактов (альтернатива - ручная разметка)
    ica = mne.preprocessing.ICA(n_components=32, random_state=97)
    ica.fit(raw_filtered)
    ica.apply(raw_filtered) 
    
    return raw_filtered

# 3. Визуализация сырых и обработанных данных
def plot_raw_vs_filtered(raw, raw_filtered, channel=0, fs = 256, duration=10):
    """
    Сравнение сырых и обработанных данных
    """
    # Выбираем канал и временной интервал
    duration = len(raw)/fs
    start, stop = raw.time_as_index([0, duration])
    data, times = raw[channel, start:stop]
    data_filt, _ = raw_filtered[channel, start:stop]
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(times, data.T)
    ax1.set_title('Raw data')
    ax1.set_ylabel('Amplitude (μV)')
    
    ax2.plot(times, data_filt.T)
    ax2.set_title('Filtered data')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

# 4. Спектральный анализ
def compute_spectrum(raw_filtered, channel=0, nperseg=256):
    """
    Вычисление спектра мощности с использованием метода Уэлча
    """
    # Получаем данные для выбранного канала
    data, times = raw_filtered[channel]
    
    # Вычисляем спектр мощности
    sfreq = raw_filtered.info['sfreq']
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    
    return freqs, psd

# 5. Визуализация спектра
def plot_spectrum(freqs, psd, l_freq=1, h_freq=40):
    """
    Визуализация спектра мощности
    """
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, 10 * np.log10(psd.T), linewidth=1)  # В dB
    plt.xlim([l_freq, h_freq])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title('EEG Power Spectrum')
    plt.grid(True)
    plt.show()

# 6. Анализ ритмов ЭЭГ
def analyze_bands(freqs, psd):
    """
    Анализ мощности в основных ритмах ЭЭГ
    """
    # Определяем границы ритмов
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    band_power = {}
    for band, (low, high) in bands.items():
        # Находим индексы частот в диапазоне
        idx = np.logical_and(freqs >= low, freqs <= high)
        # Интегрируем мощность в диапазоне
        band_power[band] = np.trapz(psd[:, idx], freqs[idx])
    
    # Нормализуем мощности относительно общей мощности
    total_power = sum(band_power.values())
    band_power_rel = {band: power/total_power for band, power in band_power.items()}
    
    return band_power, band_power_rel

# 7. Визуализация ритмов
def plot_bands(band_power_rel):
    """
    Визуализация относительной мощности ритмов
    """
    bands = list(band_power_rel.keys())
    values = list(band_power_rel.values())  # Преобразуем в обычный список
    
    print(values)  # Отладочный вывод
    print(type(values))  # Отладочный вывод
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(bands, values)  # Передаем плоский список
    plt.title('Relative Power of EEG Bands')
    plt.ylabel('Relative Power')
    
    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.show()

# Основной скрипт обработки
if __name__ == "__main__":
    # Загрузка данных (замените путь на ваш файл)
    file_path = 'data/raw_Gushchin_EEG.csv'  # Пример для .edf файла
    raw = load_eeg_data(file_path)
    
    # Предобработка
    raw_filtered = preprocess_eeg(raw)
    
    # Визуализация сырых и обработанных данных
    plot_raw_vs_filtered(raw, raw_filtered)
    
    # Спектральный анализ
    freqs, psd = compute_spectrum(raw_filtered)
    plot_spectrum(freqs, psd)
    
    # Анализ ритмов
    band_power, band_power_rel = analyze_bands(freqs, psd)
    print("Absolute band power:", band_power)
    print("Relative band power:", band_power_rel)
    
    band_power_rel = {
    "Delta": 0.1,
    "Theta": 0.2,
    "Alpha": 0.3,
    "Beta": 0.25,
    "Gamma": 0.15
    }
    # Визуализация ритмов
    plot_bands(band_power_rel)