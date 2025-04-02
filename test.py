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
        # Загрузка CSV с обработкой разных форматов чисел
        df = pd.read_csv(file_path, 
                        sep=None, 
                        engine='python',
                        decimal='.',  # Указываем десятичный разделитель
                        thousands=None,
                        encoding='utf-8',
                        dtype=str,  # Сначала читаем как строки
                        on_bad_lines='warn')
        
        # Преобразуем в числа, заменяя ошибки на NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем столбцы, которые не смогли преобразовать
        df = df.dropna(axis=1, how='all')
        
        if df.empty:
            raise ValueError("CSV файл не содержит числовых данных")
            
        # Проверяем заголовки
        if isinstance(df.columns[0], str) and df.columns[0].replace('.','',1).isdigit():
            # Если заголовки - это числа, значит настоящих заголовков нет
            ch_names = [f'EEG_{i+1}' for i in range(df.shape[1])]
            data = df.values.T
        else:
            # Используем или создаем заголовки
            ch_names = [str(col) for col in df.columns]
            data = df.values.T
        
        # Проверяем данные на NaN
        if np.isnan(data).any():
            print("Предупреждение: данные содержат пропущенные значения (NaN)")
            data = np.nan_to_num(data)  # Заменяем NaN на 0
        
        # Создаем объект RawArray
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        print(f"Успешно загружено {len(ch_names)} каналов:")
        print(ch_names)
        print(f"Форма данных: {data.shape}")
        
        return raw
    except Exception as e:
        print(f"Ошибка загрузки файла: {str(e)}")
        print("\nРекомендации:")
        print("1. Проверьте, что файл содержит только числовые данные")
        print("2. Убедитесь, что десятичный разделитель - точка (.)")
        print("3. Проверьте кодировку файла (должна быть UTF-8)")
        print("4. Удалите все нечисловые символы и заголовки, если они мешают")
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
    ica = mne.preprocessing.ICA(n_components=64, random_state=97)
    ica.fit(raw_filtered)
    ica.apply(raw_filtered)
    
    return raw_filtered

# 3. Визуализация сырых и обработанных данных
def plot_raw_vs_filtered(raw, raw_filtered, channel=0, duration=10):
    """
    Сравнение сырых и обработанных данных
    """
    # Выбираем канал и временной интервал
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
    file_path = 'data/raw_EEG_Gushchin.csv'  # Пример для .edf файла
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