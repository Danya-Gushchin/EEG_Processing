import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
import pywt
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from PyEMD import EMD
from preprocessing import denoised_signal

fs = 256
t = np.linspace(0, 1, fs, endpoint=False)

raw_data, signal_data = denoised_signal('data/raw_EEG_Gushchin.csv')

signal_data = signal_data[:,0]
raw_data = raw_data[:,0]

### 1. Фурье-спектр (FFT)
fft_spectrum = fft.fft(signal_data)
freqs = fft.fftfreq(len(signal_data), 1/fs)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(freqs[:len(freqs)//2], np.abs(fft_spectrum[:len(freqs)//2]))
plt.title("Фурье-спектр (FFT)")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")

### 2. Вейвлет-спектр (CWT)
scales = np.arange(1, 128)
cwt_coeffs, freqs_cwt = pywt.cwt(signal_data, scales, 'cmor1.5-1.0', sampling_period=1/fs)

plt.subplot(2, 2, 2)
plt.imshow(np.abs(cwt_coeffs), aspect='auto', extent=[t[0], t[-1], scales[-1], scales[0]])
plt.title("Вейвлет-спектр (CWT)")
plt.xlabel("Время (с)")
plt.ylabel("Масштаб")

### 3. Разложение на моды с использованием EMD
emd = EMD()
imfs = emd(signal_data)

### 4. Применение преобразования Гильберта к каждой моде
hilbert_spectrum = np.zeros((len(imfs), len(signal_data)))

for i, imf in enumerate(imfs):
    analytic_signal = hilbert(imf)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
    instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])  # To match the length
    hilbert_spectrum[i, :] = instantaneous_frequency

### 5. Построение спектра Гильберта-Хуанга
plt.subplot(2, 2, 3)
plt.imshow(hilbert_spectrum, aspect='auto', cmap='jet', extent=[t[0], t[-1], len(imfs), 1])
plt.title("Спектр Гильберта-Хуанга (HHS)")
plt.xlabel("Время (с)")
plt.ylabel("Моды")

### 6. Построение голо-гильбертовского спектра (Holo-Hilbert Spectrum)
# Для голо-гильбертовского спектра нужно применить EMD и преобразование Гильберта к каждой моде еще раз
# Это более сложный процесс, и здесь приведен упрощенный пример

hht_spectrum = np.zeros((len(imfs), len(signal_data)))

for i, imf in enumerate(imfs):
    emd_imf = EMD()
    imfs_imf = emd_imf(imf)
    for j, imf_imf in enumerate(imfs_imf):
        analytic_signal_imf = hilbert(imf_imf)
        instantaneous_phase_imf = np.unwrap(np.angle(analytic_signal_imf))
        instantaneous_frequency_imf = (np.diff(instantaneous_phase_imf) / (2.0*np.pi) * fs)
        instantaneous_frequency_imf = np.append(instantaneous_frequency_imf, instantaneous_frequency_imf[-1])  # To match the length
        hht_spectrum[i, :] += instantaneous_frequency_imf

plt.subplot(2, 2, 4)
plt.imshow(hht_spectrum, aspect='auto', cmap='jet', extent=[t[0], t[-1], len(imfs), 1])
plt.title("Голо-гильбертовский спектр (HHS)")
plt.xlabel("Время (с)")
plt.ylabel("Моды")

plt.tight_layout()
plt.show()