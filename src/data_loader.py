import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm


class DataLoader():
    def __init__(self):
        pass

    @staticmethod
    def read_eeg_from_csv(file_path: str | Path) -> pd.DataFrame:
        """
        Загружает EEG файл, обрабатывает возможные ошибки и возвращает 64 канала сигналов.
        Без меток, только сырой сигнал
        """
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1251']  
        
        for enc in tqdm(encodings, desc='Проверка коджировки'):
            try:
                df = pd.read_csv(
                    file_path, 
                    encoding=enc, 
                    sep=None, 
                    engine='python', 
                    index_col=False
                )
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Ошибка чтения файла: {e}")
        else:
            raise ValueError("Не удалось определить кодировку файла")
        
        print("\nФильтрация каналов EEG...")
        # Фильтрация столбцов: оставляем только 64 канала (предположительно, без мета-данных)
        signal_columns = [
            col for col in tqdm(df.columns, desc="Анализ столбцов") 
            if isinstance(col, str) and 
            not any(x in col.lower() for x in ['time', 'marker', 'event'])
        ]

        if len(signal_columns) < 64:
            raise ValueError(f"Ожидалось минимум 64 канала, но найдено {len(signal_columns)}")
        df = df[signal_columns[:64]]  # Оставляем ровно 64 сигнала
        print(f"Выбрано 64 канала")
        
        # Преобразование данных с прогресс-баром
        print("\nПреобразование данных...")
        conversion_errors: Dict[str, int] = {}
        
        for col in tqdm(df.columns, desc="Обработка каналов"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                conversion_errors[col] = na_count
                
        if conversion_errors:
            print(f"\nОбнаружены некорректные данные в {len(conversion_errors)} каналах")
            print(f"Всего строк с ошибками: {sum(conversion_errors.values())}")
            df.dropna(inplace=True)
            print("Некорректные данные удалены")
        
        print("\nЗагрузка завершена успешно!")
        print(f"Итоговый размер данных: {df.shape[0]} записей x {df.shape[1]} каналов")

        # Проверка наличия некорректных данных после преобразования
        if df.isna().sum().sum() > 0:
            print("Обнаружены некорректные данные, они будут удалены:")
            df.dropna(inplace=True)  # Удаляем строки с некорректными значениями
        
        return df
    
    @staticmethod
    def normalize_signal(signal_df: pd.DataFrame) -> np.ndarray:
        """Нормализует сигнал (вычитает среднее и делит на стандартное отклонение)"""
        if signal_df.empty:
            raise ValueError("CSV файл пуст или не содержит числовых данных.")

        # Нормализация данных (важно для нейросетей)
        eeg_data = signal_df.values
        data_mean = np.mean(eeg_data, axis=0, keepdims=True)
        data_std = np.std(eeg_data, axis=0, keepdims=True)
        eeg_data = (eeg_data - data_mean) / (data_std + 1e-8)  # Защита от деления на 0
        
        print(f"Данные после нормализации - среднее: {eeg_data.mean():.4f}, std: {eeg_data.std():.4f}")

        return eeg_data
    
if __name__=="__main__":
    loader = DataLoader()
    file_path = '/home/skill-t20/Documents/EEG_processing/data/raw_Gushchin_EEG.csv'
    data = loader.read_eeg_from_csv(file_path)
    data_norm = loader.normalize_signal(data)
    print(data_norm)
