import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from scipy.signal import butter, filtfilt

class EEGPreprocessing:
    def __init__(self, sfreq=256):
        """
        Инициализация класса предобработки ЭЭГ
        
        Parameters:
        -----------
        sfreq : float
            Частота дискретизации (по умолчанию 256 Гц)
        """
        self.sfreq = sfreq
        self.notch_filter = None
        self.bandpass_filter = None
        
    def load_eeg_data(self, file_path):
        """
        Улучшенная загрузка ЭЭГ данных из CSV с обработкой ошибок формата
        
        Parameters:
        -----------
        file_path : str
            Путь к CSV файлу с данными
        """
        try:
            # Загрузка данных с обработкой различных форматов
            df = pd.read_csv(file_path, sep=None, engine='python', decimal='.', dtype=str, on_bad_lines='skip')

            # Преобразование в числовой формат с обработкой ошибок
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Удаление полностью пустых столбцов и строк
            df = df.dropna(axis=1, how='all')
            df = df.dropna(axis=0, how='all')

            if df.empty:
                raise ValueError("CSV файл не содержит числовых данных")

            data = df.values.T

            # Удаление каналов, полностью состоящих из нулей
            non_zero_mask = ~(np.all(data == 0, axis=1))
            data = data[non_zero_mask]
            ch_names = [f"EEG_{i+1}" for i in range(data.shape[0])]

            # Замена NaN на 0 с предупреждением
            if np.isnan(data).any():
                print("Предупреждение: данные содержат NaN. Заменяем на 0.")
                data = np.nan_to_num(data)

            # Создание MNE Raw объекта
            info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)

            print(f"Успешно загружено {data.shape[0]} каналов, {data.shape[1]} отсчётов.")
            return raw

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise
    
    def design_filters(self, notch_freq=50.0, bandpass_range=(1, 40)):
        """
        Создание фильтров для предобработки
        
        Parameters:
        -----------
        notch_freq : float
            Частота для режекторного фильтра (по умолчанию 50 Гц)
        bandpass_range : tuple
            Границы полосового фильтра (low, high) в Гц
        """
        # Режекторный фильтр (notch) для 50 Гц
        nyq = 0.5 * self.sfreq
        freq = notch_freq / nyq
        b, a = butter(2, [freq-0.05, freq+0.05], btype='bandstop')
        self.notch_filter = (b, a)
        
        # Полосовой фильтр
        low = bandpass_range[0] / nyq
        high = bandpass_range[1] / nyq
        b, a = butter(4, [low, high], btype='band')
        self.bandpass_filter = (b, a)
    
    def apply_filters(self, signal):
        """
        Применение фильтров к сигналу
        
        Parameters:
        -----------
        signal : ndarray
            Входной сигнал (1D array)
        
        Returns:
        --------
        filtered_signal : ndarray
            Отфильтрованный сигнал
        """
        if self.notch_filter is None or self.bandpass_filter is None:
            raise ValueError("Фильтры не инициализированы. Сначала вызовите design_filters()")
            
        # Применяем режекторный фильтр
        signal_filtered = filtfilt(*self.notch_filter, signal)
        
        # Применяем полосовой фильтр
        signal_filtered = filtfilt(*self.bandpass_filter, signal_filtered)
        
        return signal_filtered
    
    def preprocess_pipeline(self, raw, notch_freq=50.0, bandpass_range=(1, 40), ica_n_components=15):
        """
        Полный пайплайн предобработки ЭЭГ данных
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Сырые данные ЭЭГ
        notch_freq : float
            Частота для режекторного фильтра
        bandpass_range : tuple
            Границы полосового фильтра
        ica_n_components : int
            Количество компонент для ICA
        
        Returns:
        --------
        raw_processed : mne.io.Raw
            Обработанные данные
        """
        # 1. Копируем данные, чтобы не изменять оригинал
        raw_processed = raw.copy()
        
        # 2. Создаем фильтры
        self.design_filters(notch_freq=notch_freq, bandpass_range=bandpass_range)
        
        # 3. Применяем фильтры к каждому каналу
        for i in range(len(raw_processed.ch_names)):
            raw_processed._data[i] = self.apply_filters(raw_processed._data[i])
        
        # 4. Применяем ICA для удаления артефактов (если требуется)
        if ica_n_components > 0:
            try:
                ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=97)
                ica.fit(raw_processed)
                
                # Автоматическое обнаружение артефактов
                ica.exclude = []
                
                # Обнаружение глазных артефактов
                eog_indices, eog_scores = ica.find_bads_eog(raw_processed)
                if eog_indices:
                    ica.exclude.extend(eog_indices)
                    print(f"Обнаружены и исключены {len(eog_indices)} EOG компонент")
                
                # Обнаружение мышечных артефактов
                muscle_indices = ica.find_bads_muscle(raw_processed)
                if muscle_indices:
                    ica.exclude.extend(muscle_indices)
                    print(f"Обнаружены и исключены {len(muscle_indices)} мышечных компонент")
                
                ica.apply(raw_processed)
            except Exception as e:
                print(f"Ошибка при применении ICA: {e}")
                print("Продолжаем без ICA...")
        
        return raw_processed
    
    def compare_raw_vs_processed(self, raw, processed, channel=0, fs=256):
        """
        Визуальное сравнение сырых и обработанных данных
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Сырые данные
        processed : mne.io.Raw
            Обработанные данные
        channel_idx : int
            Индекс канала для визуализации
        duration : float
            Длительность отрезка для отображения (в секундах)
        """
        # Получаем данные
        duration = raw.n_times/fs
        start, stop = raw.time_as_index([0, duration])
        raw_data, times = raw[channel, start:stop]
        proc_data, _ = processed[channel, start:stop]
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        ax1.plot(times, raw_data.T)
        ax1.set_title(f'Сырые данные (канал: {raw.ch_names[channel]})')
        ax1.set_ylabel('Амплитуда (μV)')
        ax1.grid(True)
        
        ax2.plot(times, proc_data.T)
        ax2.set_title('После предобработки')
        ax2.set_ylabel('Амплитуда (μV)')
        ax2.set_xlabel('Время (с)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Инициализация
    preprocessor = EEGPreprocessing(sfreq=256)

    # Загрузка данных
    raw = preprocessor.load_eeg_data("data/raw_Gushchin_EEG.csv")

    # Полная предобработка
    processed = preprocessor.preprocess_pipeline(
        raw,
        notch_freq=50.0,
        bandpass_range=(1, 40),
        ica_n_components=64
    )

    # Визуальное сравнение
    preprocessor.compare_raw_vs_processed(raw, processed, channel=0)