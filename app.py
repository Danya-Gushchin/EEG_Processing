import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QTabWidget, QLabel, QComboBox, QDoubleSpinBox, QCheckBox, 
                             QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class EEGProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка ЭЭГ сигналов")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.create_menu()
        self.create_tabs()
        
        self.eeg_data = None
        self.sample_rate = None
        self.ica_components = None
        
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("Файл")
        open_action = file_menu.addAction("Открыть файл")
        open_action.triggered.connect(self.open_file)
        
    def create_tabs(self):
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Первая вкладка - предобработка
        self.preprocessing_tab = QWidget()
        self.tabs.addTab(self.preprocessing_tab, "Предобработка")
        self.create_preprocessing_tab()
        
        # Вторая вкладка - визуализация спектров
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "Визуализация спектров")
        self.create_visualization_tab()
    
    def create_preprocessing_tab(self):
        layout = QVBoxLayout(self.preprocessing_tab)
        
        # Панель загрузки файла
        file_group = QGroupBox("Загрузка файла")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("Файл не выбран")
        self.load_file_btn = QPushButton("Выбрать файл")
        self.load_file_btn.clicked.connect(self.open_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.load_file_btn)
        
        # Панель фильтрации
        filter_group = QGroupBox("Фильтрация сигнала")
        filter_layout = QVBoxLayout(filter_group)
        
        # Полосовой фильтр
        bandpass_group = QGroupBox("Полосовой фильтр")
        bandpass_layout = QHBoxLayout(bandpass_group)
        
        self.bandpass_check = QCheckBox("Применить")
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 100)
        self.low_freq_spin.setValue(1.0)
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.1, 100)
        self.high_freq_spin.setValue(30.0)
        
        bandpass_layout.addWidget(self.bandpass_check)
        bandpass_layout.addWidget(QLabel("Нижняя частота (Гц):"))
        bandpass_layout.addWidget(self.low_freq_spin)
        bandpass_layout.addWidget(QLabel("Верхняя частота (Гц):"))
        bandpass_layout.addWidget(self.high_freq_spin)
        
        # Режекторный фильтр
        notch_group = QGroupBox("Режекторный фильтр")
        notch_layout = QHBoxLayout(notch_group)
        
        self.notch_check = QCheckBox("Применить")
        self.notch_freq_spin = QDoubleSpinBox()
        self.notch_freq_spin.setRange(45, 55)
        self.notch_freq_spin.setValue(50.0)
        
        notch_layout.addWidget(self.notch_check)
        notch_layout.addWidget(QLabel("Частота (Гц):"))
        notch_layout.addWidget(self.notch_freq_spin)
        
        filter_layout.addWidget(bandpass_group)
        filter_layout.addWidget(notch_group)
        
        # Панель ICA
        ica_group = QGroupBox("ICA обработка")
        ica_layout = QVBoxLayout(ica_group)
        
        # Режим ICA
        mode_layout = QHBoxLayout()
        self.auto_reject_radio = QPushButton("Авторежект")
        self.auto_reject_radio.setCheckable(True)
        self.auto_reject_radio.setChecked(True)
        self.manual_radio = QPushButton("Ручной режим")
        self.manual_radio.setCheckable(True)
        
        mode_layout.addWidget(self.auto_reject_radio)
        mode_layout.addWidget(self.manual_radio)
        
        # Кнопка расчета ICA
        self.calculate_ica_btn = QPushButton("Рассчитать ICA")
        self.calculate_ica_btn.clicked.connect(self.calculate_ica)
        
        # Область для компонентов ICA
        self.ica_scroll = QScrollArea()
        self.ica_scroll.setWidgetResizable(True)
        self.ica_components_widget = QWidget()
        self.ica_components_layout = QVBoxLayout(self.ica_components_widget)
        self.ica_scroll.setWidget(self.ica_components_widget)
        
        ica_layout.addLayout(mode_layout)
        ica_layout.addWidget(self.calculate_ica_btn)
        ica_layout.addWidget(self.ica_scroll)
        
        # График сигнала
        self.signal_figure = Figure()
        self.signal_canvas = FigureCanvas(self.signal_figure)
        
        # Добавление всех элементов на вкладку
        layout.addWidget(file_group)
        layout.addWidget(filter_group)
        layout.addWidget(ica_group)
        layout.addWidget(self.signal_canvas)
    
    def create_visualization_tab(self):
        layout = QVBoxLayout(self.visualization_tab)
        
        # Выбор метода визуализации
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод:"))
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Фурье", "Вейвлет", "Гильберта-Хуанга"])
        method_layout.addWidget(self.method_combo)
        
        self.plot_btn = QPushButton("Построить")
        self.plot_btn.clicked.connect(self.plot_spectrum)
        method_layout.addWidget(self.plot_btn)
        
        self.save_btn = QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.save_spectrum)
        method_layout.addWidget(self.save_btn)
        
        layout.addLayout(method_layout)
        
        # График спектра
        self.spectrum_figure = Figure()
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        layout.addWidget(self.spectrum_canvas)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл ЭЭГ", "", "Все файлы (*);;EDF файлы (*.edf)")
        if file_path:
            self.file_label.setText(file_path)
            # Здесь должна быть загрузка данных из файла
            # self.eeg_data, self.sample_rate = load_eeg_file(file_path)
            self.plot_original_signal()
    
    def plot_original_signal(self):
        if self.eeg_data is None:
            return
            
        self.signal_figure.clear()
        ax = self.signal_figure.add_subplot(111)
        
        # Пример отрисовки случайных данных
        t = np.arange(0, 10, 0.01)
        signal = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(len(t))
        ax.plot(t, signal)
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Амплитуда")
        ax.set_title("Исходный сигнал")
        
        self.signal_canvas.draw()
    
    def calculate_ica(self):
        if self.eeg_data is None:
            return
            
        # Здесь должна быть реализация ICA
        # self.ica_components = calculate_ica_components(self.eeg_data)
        
        # Очищаем предыдущие компоненты
        for i in reversed(range(self.ica_components_layout.count())): 
            self.ica_components_layout.itemAt(i).widget().setParent(None)
        
        # Создаем виджеты для каждого компонента (пример для 3 компонентов)
        num_components = 3  # Заменить на реальное количество компонентов
        
        for i in range(num_components):
            component_group = QGroupBox(f"Компонент {i+1}")
            component_layout = QVBoxLayout(component_group)
            
            # График компонента
            component_fig = Figure(figsize=(5, 2))
            component_ax = component_fig.add_subplot(111)
            
            # Пример данных
            t = np.arange(0, 10, 0.01)
            component_ax.plot(t, np.sin(2 * np.pi * (i+1) * t / 10))
            component_ax.set_title(f"Компонент {i+1}")
            
            component_canvas = FigureCanvas(component_fig)
            component_layout.addWidget(component_canvas)
            
            # Кнопки управления
            btn_layout = QHBoxLayout()
            
            self.reject_btn = QPushButton("Удалить")
            self.reject_btn.setCheckable(True)
            self.reject_btn.clicked.connect(lambda _, idx=i: self.toggle_component(idx))
            
            self.restore_btn = QPushButton("Восстановить")
            self.restore_btn.setCheckable(True)
            self.restore_btn.setChecked(True)
            self.restore_btn.clicked.connect(lambda _, idx=i: self.toggle_component(idx))
            
            btn_layout.addWidget(self.reject_btn)
            btn_layout.addWidget(self.restore_btn)
            component_layout.addLayout(btn_layout)
            
            self.ica_components_layout.addWidget(component_group)
        
        self.plot_processed_signal()
    
    def toggle_component(self, component_idx):
        # Здесь должна быть логика удаления/восстановления компонента
        print(f"Компонент {component_idx} изменен")
        self.plot_processed_signal()
    
    def plot_processed_signal(self):
        if self.eeg_data is None:
            return
            
        self.signal_figure.clear()
        ax = self.signal_figure.add_subplot(111)
        
        # Пример отрисовки обработанного сигнала
        t = np.arange(0, 10, 0.01)
        processed_signal = np.sin(2 * np.pi * t)  # Здесь должен быть реальный обработанный сигнал
        ax.plot(t, processed_signal)
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Амплитуда")
        ax.set_title("Обработанный сигнал")
        
        self.signal_canvas.draw()
    
    def plot_spectrum(self):
        if self.eeg_data is None:
            return
            
        method = self.method_combo.currentText()
        self.spectrum_figure.clear()
        ax = self.spectrum_figure.add_subplot(111)
        
        # Пример отрисовки спектра
        f = np.linspace(0, 50, 1000)
        if method == "Фурье":
            spectrum = np.exp(-(f-10)**2/10) + 0.5*np.exp(-(f-25)**2/10)
            ax.plot(f, spectrum)
        elif method == "Вейвлет":
            spectrum = np.exp(-(f-15)**2/10) + 0.7*np.exp(-(f-30)**2/10)
            ax.plot(f, spectrum)
        else:  # Гильберта-Хуанга
            spectrum = np.exp(-(f-20)**2/15) + 0.3*np.exp(-(f-35)**2/15)
            ax.plot(f, spectrum)
        
        ax.set_xlabel("Частота (Гц)")
        ax.set_ylabel("Мощность")
        ax.set_title(f"Спектр ({method})")
        
        self.spectrum_canvas.draw()
    
    def save_spectrum(self):
        if self.eeg_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить спектр", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)")
        if file_path:
            self.spectrum_figure.savefig(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGProcessingApp()
    window.show()
    sys.exit(app.exec())