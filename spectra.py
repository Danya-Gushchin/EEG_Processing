import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt
from PyEMD import EMD
from test import EEGPreprocessing

class EEGSpectralAnalyzer:
    def __init__(self, eeg_data, sampling_rate, channels_names=None):
        """
        eeg_data: ndarray (channels x time)
        sampling_rate: int (Hz)
        channels_names: list of str (optional)
        """
        self.eeg_data = eeg_data
        self.fs = sampling_rate
        self.n_channels, self.n_samples = eeg_data.shape
        self.channels_names = channels_names or [f"Channel {i}" for i in range(self.n_channels)]
        
        # # Фильтр для удаления сетевой наводки 50 Гц
        # self.design_notch_filter(notch_freq=50.0, Q=30)

    # def design_notch_filter(self, notch_freq=50.0, Q=30):
    #     """Создание режекторного фильтра"""
    #     nyq = 0.5 * self.fs
    #     freq = notch_freq / nyq
    #     b, a = butter(2, [freq-0.05, freq+0.05], btype='bandstop')
    #     self.notch_filter = (b, a)

    # def apply_notch_filter(self, signal):
    #     """Применение режекторного фильтра"""
    #     return filtfilt(*self.notch_filter, signal)

    def compute_fft(self, channel):
        signal = self.eeg_data[channel]
        freqs = fftfreq(self.n_samples, 1/self.fs)
        spectrum = np.abs(fft(signal))
        
        # Нормализация спектра
        spectrum = spectrum / self.n_samples
        return freqs[:self.n_samples // 2], spectrum[:self.n_samples // 2]

    def compute_wavelet(self, channel, wavelet='morl', scales=None):
        signal = self.eeg_data[channel]
        
        if scales is None:
            scales = np.arange(1, 128)
        
        coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1/self.fs)
        power = np.abs(coeffs) ** 2
        
        # Улучшенная нормализация
        power = (power - np.min(power)) / (np.max(power) - np.min(power) + 1e-12)
        return freqs, power

    def compute_hht(self, channel):
        signal = self.eeg_data[channel]
       
        emd = EMD()
        imfs = emd(signal)
        
        # Подготовка данных для тепловой карты
        time_points = np.arange(self.n_samples) / self.fs
        freq_bins = np.linspace(0, self.fs/2, 200)  # Увеличили разрешение
        hht_spectrum = np.zeros((len(freq_bins)-1, len(time_points)))
        
        for imf in imfs:
            analytic = hilbert(imf)
            inst_amp = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) * self.fs / (2.0 * np.pi)
            inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])
            
            for t in range(len(time_points)):
                freq = inst_freq[t]
                if 0.5 < freq < self.fs/2:  # Игнорируем очень низкие частоты
                    bin_idx = np.digitize(freq, freq_bins) - 1
                    if 0 <= bin_idx < len(hht_spectrum):
                        hht_spectrum[bin_idx, t] += inst_amp[t]
        
        # Нормализация HHT спектра
        hht_spectrum = np.log1p(hht_spectrum)  # Логарифмическое масштабирование
        return time_points, freq_bins[:-1], hht_spectrum

    def plot_spectrum(self, channel, method='fft'):
        plt.figure(figsize=(12, 7))
        
        if method == 'fft':
            freqs, spectrum = self.compute_fft(channel)
            plt.plot(freqs, spectrum)
            plt.title(f"FFT Spectrum - {self.channels_names[channel]}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.xlim(0, self.fs/2)
            plt.grid(True)
            
            # Подсветка 50 Гц
            plt.axvline(x=50, color='r', linestyle='--', alpha=0.3)
            plt.text(50, np.max(spectrum)*0.9, '50 Hz', color='r')

        elif method == 'wavelet':
            freqs, power = self.compute_wavelet(channel)
            plt.imshow(power, extent=[0, self.n_samples/self.fs, freqs[-1], freqs[0]],
                      aspect='auto', cmap='viridis', interpolation='bilinear')
            plt.title(f"Wavelet Power Spectrum - {self.channels_names[channel]}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label='Normalized Power')
            plt.clim(0, 1)  # Фиксированный диапазон цветов

        elif method == 'hht':
            time_points, freq_bins, hht_spectrum = self.compute_hht(channel)
            plt.imshow(hht_spectrum, extent=[0, self.n_samples/self.fs, freq_bins[-1], freq_bins[0]],
                      aspect='auto', cmap='plasma', interpolation='gaussian')
            plt.title(f"Hilbert-Huang Spectrum - {self.channels_names[channel]}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label='Log Amplitude')
            
        else:
            raise ValueError("Unknown method. Use 'fft', 'wavelet', or 'hht'.")
        
        plt.tight_layout()
        plt.show()

    # Дополнительные методы для анализа центров принятия решений
    def compute_erp(self, event_times, window=(-0.2, 0.8)):
        """
        Вычисление потенциалов, связанных с событиями (ERP)
        event_times: список времен событий в секундах
        window: кортеж (pre, post) в секундах
        """
        n_samples_pre = int(abs(window[0]) * self.fs)
        n_samples_post = int(window[1] * self.fs)
        erp = np.zeros((self.n_channels, n_samples_pre + n_samples_post))
        
        for time in event_times:
            start_idx = int(time * self.fs) - n_samples_pre
            end_idx = start_idx + n_samples_pre + n_samples_post
            
            if start_idx >= 0 and end_idx <= self.n_samples:
                erp += self.eeg_data[:, start_idx:end_idx]
        
        return erp / len(event_times)

    def compute_time_frequency_power(self, channel, freq_bands=None):
        """
        Вычисление мощности в частотных диапазонах во времени
        freq_bands: словарь {название: (f_min, f_max)}
        """
        if freq_bands is None:
            freq_bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 80)
            }
        
        _, freqs, power = self.compute_hht(channel)
        time_points = np.arange(self.n_samples) / self.fs
        band_power = {}
        
        for band, (f_min, f_max) in freq_bands.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            band_power[band] = np.sum(power[mask, :], axis=0)
        
        return time_points, band_power

    def plot_topographic_map(self, values, title=""):
        """
        Простая топографическая карта (заглушка - нужны реальные координаты электродов)
        values: массив значений для каждого канала
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(np.random.rand(self.n_channels), np.random.rand(self.n_channels), 
                   c=values, s=200, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(title)
        
        for i, name in enumerate(self.channels_names):
            plt.text(np.random.rand(), np.random.rand(), name, fontsize=8)
        
        plt.axis('off')
        plt.show()

# Пример использования
eeg_data_path = "data/raw_Gushchin_EEG.csv"
preprocessor = EEGPreprocessing(sfreq=256)
raw = preprocessor.load_eeg_data(eeg_data_path)
processed = preprocessor.preprocess_pipeline(
    raw,
    notch_freq=50.0,
    bandpass_range=(20, 80),
    ica_n_components=0
)

eeg_data_processed = processed.get_data()  # mne object --> ndarray

# Имена каналов (пример)
channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']

analyzer = EEGSpectralAnalyzer(eeg_data_processed, sampling_rate=256, channels_names=channel_names)

# Анализ спектров
analyzer.plot_spectrum(channel=0, method='fft')
analyzer.plot_spectrum(channel=0, method='wavelet')
analyzer.plot_spectrum(channel=0, method='hht')

# Дополнительный анализ для локализации центров принятия решений
# 1. Анализ ERP (нужны времена событий)
# event_times = [...]  # времена предъявления стимулов
# erp = analyzer.compute_erp(event_times)
# analyzer.plot_topographic_map(erp.max(axis=1), "Max ERP Amplitude")

# 2. Анализ мощности в частотных диапазонах
time_points, band_power = analyzer.compute_time_frequency_power(channel=0)
plt.figure(figsize=(12, 6))
for band, power in band_power.items():
    plt.plot(time_points, power, label=band)
plt.title("Band Power Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.show()