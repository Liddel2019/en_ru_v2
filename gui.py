import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
import threading
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from colorama import init, Fore, Style
from autotest import AutoTest

init()

class GUI:
    def __init__(self, root, trainer, config, data_processor):
        """
        Инициализация графического интерфейса для переводчика EN-RU.
        """
        self.root = root
        self.root.title("EN-RU Translator")
        self.config = config
        self.data_processor = data_processor
        self.trainer = trainer
        self.progress = tk.DoubleVar()
        self.download_queue = queue.Queue()
        self.metric_queue = queue.Queue()
        self.training_info = tk.StringVar(value="Эпоха: 0/0, Шаг: 0/0, Всего шагов: 0/0")
        self.runs = {}  # Словарь для хранения метрик
        self.current_run_id = None
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        print(f"===GUI.py===\n{Fore.BLUE}Инициализация интерфейса...{Style.RESET_ALL}")

        # Настройка окна
        self.root.resizable(True, True)
        self.root.minsize(800, 600)
        self.root.geometry("1000x700")

        # Создание главного фрейма
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        # Создание вкладок
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)

        # Создание фреймов для вкладок
        self.training_frame = ttk.Frame(self.notebook)
        self.datasets_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.translation_frame = ttk.Frame(self.notebook)
        self.logs_frame = ttk.Frame(self.notebook)
        self.autotest_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.training_frame, text="Обучение")
        self.notebook.add(self.datasets_frame, text="Датасеты")
        self.notebook.add(self.metrics_frame, text="Метрики")
        self.notebook.add(self.translation_frame, text="Перевод")
        self.notebook.add(self.logs_frame, text="Логи")
        self.notebook.add(self.autotest_frame, text="Автотест")

        # Инициализация вкладок
        self.create_training_tab()
        self.create_datasets_tab()
        self.create_metrics_tab()
        self.create_translation_tab()
        self.create_logs_tab()
        self.create_autotest_tab()

        # Автоматическое обновление списка датасетов
        self.update_dataset_list()

        # Запуск проверки очередей
        self.check_download_queue()
        self.check_metric_queue()
        print(f"===GUI.py===\n{Fore.GREEN}Интерфейс успешно инициализирован{Style.RESET_ALL}")

    def create_training_tab(self):
        """
        Создание вкладки обучения.
        """
        self.training_info_label = ttk.Label(self.training_frame, textvariable=self.training_info, font=("Arial", 12))
        self.training_info_label.pack(padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(self.training_frame, variable=self.progress, maximum=100, style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(padx=10, pady=5, fill="x")
        self.progress_label = ttk.Label(self.training_frame, text="Прогресс: 0%", font=("Arial", 10))
        self.progress_label.pack(padx=10, pady=5)

        self.settings_frame = ttk.LabelFrame(self.training_frame, text="Параметры обучения", padding="10")
        self.settings_frame.pack(padx=10, pady=10, fill="x")
        self.create_settings_fields()

        self.dataset_frame = ttk.LabelFrame(self.training_frame, text="Выбор датасетов", padding="10")
        self.dataset_frame.pack(padx=10, pady=10, fill="x")
        self.dataset_list = {}

        self.filter_frame = ttk.LabelFrame(self.training_frame, text="Фильтры длины предложений", padding="10")
        self.filter_frame.pack(padx=10, pady=10, fill="x")
        self.create_filter_fields()

        button_frame = ttk.Frame(self.training_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Обновить список датасетов", command=self.update_dataset_list, style="TButton").pack(side="left", padx=5)
        ttk.Button(button_frame, text="Начать обучение", command=self.start_training, style="TButton").pack(side="left", padx=5)
        ttk.Button(button_frame, text="Остановить обучение", command=self.stop_training, style="TButton").pack(side="left", padx=5)

    def create_datasets_tab(self):
        """
        Создание вкладки для управления датасетами.
        """
        self.available_datasets_frame = ttk.LabelFrame(self.datasets_frame, text="Доступные датасеты", padding="10")
        self.available_datasets_frame.pack(padx=10, pady=10, fill="x")

        self.download_frame = ttk.LabelFrame(self.datasets_frame, text="Датасеты для загрузки", padding="10")
        self.download_frame.pack(padx=10, pady=10, fill="x")
        self.download_list = {}

    def create_metrics_tab(self):
        """
        Создание вкладки метрик.
        """
        metric_options = ["Потери на обучении", "Потери на валидации", "BLEU Score", "Время обучения (сек)"]
        self.selected_metric = tk.StringVar(value=metric_options[0])
        self.metric_dropdown = ttk.OptionMenu(self.metrics_frame, self.selected_metric, *metric_options, command=lambda _: self.update_plot())
        self.metric_dropdown.pack(pady=5)

        self.test_selection_frame = ttk.LabelFrame(self.metrics_frame, text="Выбор тестов для сравнения", padding="10")
        self.test_selection_frame.pack(padx=10, pady=5, fill="x")
        self.test_vars = {}
        self.update_test_selection()

        self.metric_labels_frame = ttk.Frame(self.metrics_frame)
        self.metric_labels_frame.pack(pady=5)
        self.train_loss_label = ttk.Label(self.metric_labels_frame, text="Потери на обучении: N/A", font=("Arial", 10))
        self.val_loss_label = ttk.Label(self.metric_labels_frame, text="Потери на валидации: N/A", font=("Arial", 10))
        self.bleu_label = ttk.Label(self.metric_labels_frame, text="BLEU Score: N/A", font=("Arial", 10))
        self.time_label = ttk.Label(self.metric_labels_frame, text="Время обучения: N/A", font=("Arial", 10))
        self.train_loss_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.val_loss_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.bleu_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.time_label.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.metrics_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_translation_tab(self):
        """
        Создание вкладки для перевода текста.
        """
        self.text_entry = ttk.Entry(self.translation_frame, width=50, font=("Arial", 12))
        self.text_entry.pack(padx=10, pady=10, fill="x")

        ttk.Button(self.translation_frame, text="Перевести", command=self.translate, style="TButton").pack(pady=5)

        self.translation_result = ttk.Label(self.translation_frame, text="", font=("Arial", 12), wraplength=600)
        self.translation_result.pack(padx=10, pady=10, fill="x")

    def create_logs_tab(self):
        """
        Создание вкладки для отображения логов.
        """
        self.log_text = scrolledtext.ScrolledText(self.logs_frame, height=20, font=("Arial", 10), state='disabled')
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)
        ttk.Button(self.logs_frame, text="Очистить логи", command=self.clear_logs, style="TButton").pack(pady=5)

    def create_autotest_tab(self):
        """
        Создание вкладки для автотеста.
        """
        self.autotest_settings_frame = ttk.LabelFrame(self.autotest_frame, text="Параметры автотеста", padding="10")
        self.autotest_settings_frame.pack(padx=10, pady=10, fill="x")

        autotest_settings = [
            ("n_heads_min", "Мин. число голов", 1, 16, 1, 0),
            ("n_heads_max", "Макс. число голов", 1, 16, 1, 0),
            ("n_layers_min", "Мин. число слоев", 2, 6, 1, 0),
            ("n_layers_max", "Макс. число слоев", 2, 6, 1, 0),
            ("d_model_min", "Мин. размер модели", 64, 512, 8, 0),
            ("d_model_max", "Макс. размер модели", 64, 512, 8, 0),
            ("n_calls", "Количество тестов", 10, 100, 1, 0),
        ]
        self.autotest_vars = {}
        for i, (param, label, min_val, max_val, increment, decimals) in enumerate(autotest_settings):
            frame = ttk.Frame(self.autotest_settings_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            ttk.Label(frame, text=label, font=("Arial", 10)).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value=str(min_val if "min" in param else max_val if param != "n_calls" else "100"))
            self.autotest_vars[param] = var
            spinbox = ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, increment=increment,
                                  width=15, formatმო�

        self.autotest_settings_frame.columnconfigure(0, weight=1)

        # Поле для выбора метрики и оптимизации
        metric_frame = ttk.Frame(self.autotest_settings_frame)
        metric_frame.grid(row=len(autotest_settings), column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(metric_frame, text="Метрика для оптимизации", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.optimize_metric = tk.StringVar(value="BLEU Score")
        metric_dropdown = ttk.OptionMenu(metric_frame, self.optimize_metric, "BLEU Score", "BLEU Score", "Train Loss", "Val Loss")
        metric_dropdown.grid(row=0, column=1, sticky="e")
        self.optimize_direction = tk.StringVar(value="Максимизировать")
        direction_dropdown = ttk.OptionMenu(metric_frame, self.optimize_direction, "Максимизировать", "Максимизировать", "Минимизировать")
        direction_dropdown.grid(row=0, column=2, sticky="e", padx=5)

        ttk.Button(self.autotest_frame, text="Запустить автотест", command=self.start_autotest, style="TButton").pack(pady=10)

        self.autotest_results_frame = ttk.LabelFrame(self.autotest_frame, text="Результаты автотеста", padding="10")
        self.autotest_results_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.autotest_tree = ttk.Treeview(self.autotest_results_frame, columns=("Run ID", "n_heads", "n_layers", "d_model", "Train Loss", "Val Loss", "BLEU"), show="headings")
        self.autotest_tree.heading("Run ID", text="ID теста")
        self.autotest_tree.heading("n_heads", text="Число голов")
        self.autotest_tree.heading("n_layers", text="Число слоев")
        self.autotest_tree.heading("d_model", text="Размер модели")
        self.autotest_tree.heading("Train Loss", text="Потери на обучении")
        self.autotest_tree.heading("Val Loss", text="Потери на валидации")
        self.autotest_tree.heading("BLEU", text="BLEU Score")
        self.autotestmanagement_frame = ttk.Frame(self.autotest_results_frame)
        self.autotestmanagement_frame.pack(fill='x', pady=5)
        ttk.Button(self.autotestmanagement_frame, text="Очистить результаты", command=self.clear_autotest_results, style="TButton").pack(side="left", padx=5)
        ttk.Button(self.autotestmanagement_frame, text="Обновить матрицы", command=self.update_matrices, style="TButton").pack(side="left", padx=5)
        self.autotest_tree.pack(fill="both", expand=True)

        self.matrices_frame = ttk.LabelFrame(self.autotest_frame, text="Матрицы анализа", padding="10")
        self.matrices_frame.pack(padx=10, pady=10, fill="x")
        self.correlation_text = scrolledtext.ScrolledText(self.matrices_frame, height=5, font=("Arial", 10), state='disabled')
        self.correlation_text.pack(padx=5, pady=5, fill="x")
        self.regression_text = scrolledtext.ScrolledText(self.matrices_frame, height=5, font=("Arial", 10), state='disabled')
        self.regression_text.pack(padx=5, pady=5, fill="x")

    def create_settings_fields(self):
        """
        Создание полей настроек.
        """
        settings = [
            ("batch_size", "Размер пакета", 1, 1024, 1, 0),
            ("epochs", "Количество эпох", 1, 1000000, 1, 0),
            ("learning_rate", "Скорость обучения", 0.00001, 1.0, 0.00001, 5),
            ("max_len", "Максимальная длина", 1, 100, 1, 0),
            ("d_model", "Размер модели", 8, 1024, 8, 0),
            ("n_heads", "Количество голов", 1, 16, 1, 0),
            ("n_layers", "Количество слоев", 1, 12, 1, 0),
            ("dropout", "Dropout", 0.0, 0.999, 0.01, 2),
            ("val_split", "Доля валидации", 0.0, 1.0, 0.01, 2),
            ("min_chars", "Мин. длина предложения", 1, 100, 1, 0),
            ("max_chars", "Макс. длина предложения", 1, 500, 10, 0),
        ]
        self.settings_vars = {}
        for i, (param, label, min_val, max_val, increment, decimals) in enumerate(settings):
            frame = ttk.Frame(self.settings_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            ttk.Label(frame, text=label, font=("Arial", 10)).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value=str(getattr(self.config, param)))
            self.settings_vars[param] = var
            spinbox = ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, increment=increment,
                                  width=15, format=f"%.{decimals}f" if decimals > 0 else "%d", font=("Arial", 10))
            spinbox.grid(row=0, column=1, sticky="e")
            spinbox.bind("<KeyRelease>", lambda event, p=param, min_v=min_val, max_v=max_val: self.validate_spinbox_input(event, p, min_v, max_v))
            spinbox.bind("<<Increment>>", lambda event, p=param, min_v=min_val, max_v=max_val: self.validate_spinbox_input(event, p, min_v, max_v))
            spinbox.bind("<<Decrement>>", lambda event, p=param, min_v=min_val, max_v=max_val: self.validate_spinbox_input(event, p, min_v, max_v))
        self.settings_frame.columnconfigure(0, weight=1)

    def create_filter_fields(self):
        """
        Создание полей для фильтрации по длине предложений.
        """
        self.length_categories = {
            "Короткие": (1, 50),
            "Средние": (51, 100),
            "Длинные": (101, 200),
            "Очень длинные": (201, 500)
        }
        self.filter_vars = {}
        for i, (category, (min_len, max_len)) in enumerate(self.length_categories.items()):
            frame = ttk.Frame(self.filter_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            var = tk.BooleanVar(value=True)
            self.filter_vars[category] = var
            ttk.Checkbutton(frame, text=f"{category} предложения ({min_len}-{max_len} символов)", variable=var, style="TCheckbutton").grid(row=0, column=0, sticky="w")
        self.filter_frame.columnconfigure(0, weight=1)

    def validate_spinbox_input(self, event, param, min_val, max_val):
        """
        Валидация ввода для Spinbox.
        """
        value = event.widget.get()
        if not value or value == '-':
            return True
        try:
            value = value.strip()
            if value.startswith('-') and len(value) > 1:
                return False
            val = float(value) if param in ['learning_rate', 'dropout', 'val_split'] else int(value)
            min_val = float(min_val) if '.' in str(min_val) else int(min_val)
            max_val = float(max_val) if '.' in str(max_val) else int(max_val)
            if min_val <= val <= max_val:
                if hasattr(self.config, param):
                    setattr(self.config, param, val)
                    self.log_message(f"Обновлен параметр: {param}={val}")
                return True
            self.log_message(f"Ошибка: Значение {param}={value} вне диапазона [{min_val}, {max_val}]", Fore.RED)
            event.widget.delete(0, tk.END)
            event.widget.insert(0, str(min_val if "min" in param else max_val))
            return False
        except ValueError:
            self.log_message(f"Ошибка: Неверный ввод для {param}: {value}", Fore.RED)
            event.widget.delete(0, tk.END)
            event.widget.insert(0, str(min_val if "min" in param else max_val))
            return False

    def update_dataset_list(self):
        """
        Обновление списка датасетов.
        """
        try:
            datasets_info = self.data_processor.check_datasets()
            for widget in self.available_datasets_frame.winfo_children():
                widget.destroy()
            for name, info in datasets_info.items():
                var = tk.BooleanVar(value=info["valid"])
                chk = ttk.Checkbutton(self.available_datasets_frame, text=f"{name} ({info['num_rows']} строк, {info['reason']})",
                                      variable=var, style="TCheckbutton")
                chk.pack(side="top", fill="x", pady=2)
                self.dataset_list[name] = {"var": var, "info": info}
                self.log_message(f"Добавлен датасет: {name}")

            for widget in self.download_frame.winfo_children():
                widget.destroy()
            for name, info in datasets_info.items():
                if "url" in info:
                    btn = ttk.Button(self.download_frame, text=f"Скачать {name}",
                                     command=lambda n=name, u=info["url"]: self.start_download(n, u), style="TButton")
                    btn.pack(side="top", fill="x", pady=2)
                    self.download_list[name] = btn
        except Exception as e:
            self.log_message(f"Ошибка при обновлении списка датасетов: {str(e)}", Fore.RED)

    def start_download(self, dataset_name, url):
        """
        Запуск загрузки датасета.
        """
        def download_thread():
            try:
                def update_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        progress = (block_num * block_size) / total_size * 100
                        self.progress.set(progress)
                        self.progress_label.config(text=f"Прогресс: {progress:.1f}%")
                success, message = self.trainer.data_processor.download_dataset(dataset_name, url, update_progress)
                self.download_queue.put((dataset_name, success, message))
            except Exception as e:
                self.download_queue.put((dataset_name, False, f"Ошибка загрузки: {str(e)}"))

        self.log_message(f"Начало загрузки датасета: {dataset_name}")
        threading.Thread(target=download_thread, daemon=True).start()

    def check_download_queue(self):
        """
        Проверка очереди загрузок.
        """
        try:
            while True:
                dataset_name, success, message = self.download_queue.get_nowait()
                if success:
                    messagebox.showinfo("Успех", f"Датасет {dataset_name} успешно загружен: {message}")
                    self.log_message(f"Датасет {dataset_name} успешно загружен: {message}", Fore.GREEN)
                else:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить датасет {dataset_name}: {message}")
                    self.log_message(f"Ошибка загрузки {dataset_name}: {message}", Fore.RED)
                self.update_dataset_list()
        except queue.Empty:
            pass
        self.root.after(100, self.check_download_queue)

    def check_metric_queue(self):
        """
        Проверка очереди метрик.
        """
        try:
            while True:
                metrics = self.metric_queue.get_nowait()
                if self.current_run_id not in self.runs:
                    self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': [], 'time': []}
                self.runs[self.current_run_id]['train_loss'].append(metrics['train_loss'])
                self.runs[self.current_run_id]['val_loss'].append(metrics['val_loss'])
                self.runs[self.current_run_id]['bleu'].append(metrics['bleu'])
                self.runs[self.current_run_id]['time'].append(metrics.get('time', 0))
                self.train_loss_label.config(text=f"Потери на обучении: {metrics['train_loss']:.4f}")
                self.val_loss_label.config(text=f"Потери на валидации: {metrics['val_loss']:.4f}")
                self.bleu_label.config(text=f"BLEU Score: {metrics['bleu']:.4f}")
                self.time_label.config(text=f"Время обучения: {metrics.get('time', 0):.2f} сек")
                self.update_test_selection()
                self.update_plot()
                self.log_message(f"Обновлены метрики: Потери (обучение: {metrics['train_loss']:.4f}, валидация: {metrics['val_loss']:.4f}), BLEU: {metrics['bleu']:.4f}, Время: {metrics.get('time', 0):.2f} сек")
        except queue.Empty:
            pass
        self.root.after(100, self.check_metric_queue)

    def update_test_selection(self):
        """
        Обновление списка тестов для сравнения.
        """
        for widget in self.test_selection_frame.winfo_children():
            widget.destroy()
        for run_id in self.runs:
            var = tk.BooleanVar(value=False)
            self.test_vars[run_id] = var
            ttk.Checkbutton(self.test_selection_frame, text=f"Тест {run_id}",
                            variable=var, command=self.update_plot, style="TCheckbutton").pack(side="top", fill="x", pady=2)

    def update_plot(self):
        """
        Обновление графика.
        """
        self.ax.clear()
        metric = self.selected_metric.get()
        metric_key = {
            'Потери на обучении': 'train_loss',
            'Потери на валидации': 'val_loss',
            'BLEU Score': 'bleu',
            'Время обучения (сек)': 'time'
        }[metric]
        for idx, (run_id, run_data) in enumerate(self.runs.items()):
            if self.test_vars.get(run_id, tk.BooleanVar(value=False)).get() and run_data[metric_key]:
                self.ax.plot(range(1, len(run_data[metric_key]) + 1), run_data[metric_key], label=f"Тест {run_id}", color=self.colors[idx % len(self.colors)])
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel(metric)
        self.ax.set_title(f"{metric} по эпохам")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def get_selected_filters(self):
        """
        Получение выбранных фильтров длины предложений.
        """
        return {category: (min_len, max_len) for category, (min_len, max_len) in self.length_categories.items() if self.filter_vars[category].get()}

    def start_training(self):
        """
        Запуск обучения модели.
        """
        selected_datasets = [name for name, info in self.dataset_list.items() if info["var"].get() and info["info"]["valid"]]
        if not selected_datasets:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один действительный датасет!")
            self.log_message("Предупреждение: Не выбран ни один действительный датасет", Fore.YELLOW)
            return
        self.config.dataset_paths = [os.path.join(self.config.datasets_dir, name) for name in selected_datasets]
        self.data_processor.length_filters = self.get_selected_filters()
        if not self.data_processor.length_filters:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один фильтр длины предложений!")
            self.log_message("Предупреждение: Не выбран ни один фильтр длины предложений", Fore.YELLOW)
            return

        self.current_run_id = int(datetime.datetime.now().timestamp())
        self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': [], 'time': []}
        self.train_loss_label.config(text="Потери на обучении: N/A")
        self.val_loss_label.config(text="Потери на валидации: N/A")
        self.bleu_label.config(text="BLEU Score: N/A")
        self.time_label.config(text="Время обучения: N/A")
        self.progress.set(0)
        self.progress_label.config(text="Прогресс: 0%")
        self.training_info.set("Эпоха: 0/0, Шаг: 0/0, Всего шагов: 0/0")
        self.log_message("Сброс прогресса перед началом обучения", Fore.YELLOW)

        run_dir = f"runs/run_{self.current_run_id}"
        os.makedirs(run_dir, exist_ok=True)
        self.config.model_path = f"{run_dir}/model.pt"

        with open(f"{run_dir}/config.txt", 'w', encoding='utf-8') as f:
            for attr, value in vars(self.config).items():
                f.write(f"{attr}: {value}\n")
        self.log_message(f"Конфигурация сохранена в {run_dir}/config.txt", Fore.GREEN)

        try:
            self.trainer.model = self.trainer.model.__class__(self.config)
            self.trainer.model.to(self.trainer.device)
        except Exception as e:
            self.log_message(f"Ошибка инициализации модели: {str(e)}", Fore.RED)
            return

        def training_thread():
            try:
                self.log_message("Вызов метода Trainer.train", Fore.BLUE)
                self.trainer.train(self.update_training_progress, self.metric_queue, self.current_run_id)
                if self.current_run_id in self.runs:
                    metrics_df = pd.DataFrame(self.runs[self.current_run_id])
                    metrics_df.to_csv(f"{run_dir}/metrics.csv", index_label='epoch')
                    self.log_message(f"Метрики сохранены в {run_dir}/metrics.csv", Fore.GREEN)

                    for metric, key in [('Потери на обучении', 'train_loss'), ('Потери на валидации', 'val_loss'), ('BLEU Score', 'bleu'), ('Время обучения', 'time')]:
                        plt.figure()
                        plt.plot(range(1, len(self.runs[self.current_run_id][key]) + 1), self.runs[self.current_run_id][key], label=metric, color=self.colors[0])
                        plt.xlabel("Эпоха")
                        plt.ylabel(metric)
                        plt.title(f"{metric} для теста {self.current_run_id}")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f"{run_dir}/{key}.png")
                        plt.close()
                    self.log_message(f"Графики сохранены в {run_dir}", Fore.GREEN)
            except Exception as e:
                self.log_message(f"Ошибка обучения: {str(e)}", Fore.RED)

        self.log_message("Начало обучения модели")
        threading.Thread(target=training_thread, daemon=True).start()

    def start_autotest(self):
        """
        Запуск автотеста через AutoTest класс.
        """
        selected_datasets = [name for name, info in self.dataset_list.items() if info["var"].get() and info["info"]["valid"]]
        if not selected_datasets:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один действительный датасет!")
            self.log_message("Предупреждение: Не выбран ни один действительный датасет", Fore.YELLOW)
            return
        self.config.dataset_paths = [os.path.join(self.config.datasets_dir, name) for name in selected_datasets]
        self.data_processor.length_filters = self.get_selected_filters()
        if not self.data_processor.length_filters:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один фильтр длины предложений!")
            self.log_message("Предупреждение: Не выбран ни один фильтр длины предложений", Fore.YELLOW)
            return

        try:
            n_heads_min = int(self.autotest_vars["n_heads_min"].get())
            n_heads_max = int(self.autotest_vars["n_heads_max"].get())
            n_layers_min = int(self.autotest_vars["n_layers_min"].get())
            n_layers_max = int(self.autotest_vars["n_layers_max"].get())
            d_model_min = int(self.autotest_vars["d_model_min"].get())
            d_model_max = int(self.autotest_vars["d_model_max"].get())
            n_calls = int(self.autotest_vars["n_calls"].get())
            optimize_metric = self.optimize_metric.get()
            optimize_direction = self.optimize_direction.get()
        except ValueError as e:
            messagebox.showerror("Ошибка", "Неверные значения параметров автотеста!")
            self.log_message(f"Ошибка: Неверные значения параметров автотеста: {str(e)}", Fore.RED)
            return

        if n_heads_min > n_heads_max or n_layers_min > n_layers_max or d_model_min > d_model_max:
            messagebox.showerror("Ошибка", "Минимальные значения не могут быть больше максимальных!")
            self.log_message("Ошибка: Неверные диапазоны гиперпараметров", Fore.RED)
            return
        if d_model_min % 8 != 0 or d_model_max % 8 != 0:
            messagebox.showerror("Ошибка", "Размер модели должен быть кратен 8!")
            self.log_message("Ошибка: Размер модели должен быть кратен 8", Fore.RED)
            return

        autotest = AutoTest(self.trainer, self.config, self.data_processor, self.runs, self.autotest_tree, self.log_message, self.update_matrices, self.metric_queue)
        autotest.start(n_heads_min, n_heads_max, n_layers_min, n_layers_max, d_model_min, d_model_max, n_calls, optimize_metric, optimize_direction)

    def update_autotest_results(self):
        """
        Обновление таблицы результатов автотеста.
        """
        try:
            for item in self.autotest_tree.get_children():
                self.autotest_tree.delete(item)
            for result in AutoTest.results:
                self.autotest_tree.insert("", "end", values=(
                    result['run_id'],
                    result['n_heads'],
                    result['n_layers'],
                    result['d_model'],
                    f"{result['train_loss']:.4f}" if result['train_loss'] != float('inf') else "N/A",
                    f"{result['val_loss']:.4f}" if result['val_loss'] != float('inf') else "N/A",
                    f"{result['bleu']:.4f}"
                ))
            self.log_message("Таблица результатов автотеста обновлена")
        except Exception as e:
            self.log_message(f"Ошибка обновления таблицы результатов: {str(e)}", Fore.RED)

    def clear_autotest_results(self):
        """
        Очистка результатов автотеста.
        """
        try:
            AutoTest.results = []
            for item in self.autotest_tree.get_children():
                self.autotest_tree.delete(item)
            self.correlation_text.config(state='normal')
            self.correlation_text.delete(1.0, tk.END)
            self.correlation_text.config(state='disabled')
            self.regression_text.config(state='normal')
            self.regression_text.delete(1.0, tk.END)
            self.regression_text.config(state='disabled')
            self.log_message("Результаты автотеста очищены", Fore.YELLOW)
        except Exception as e:
            self.log_message(f"Ошибка очистки результатов автотеста: {str(e)}", Fore.RED)

    def update_matrices(self):
        """
        Обновление корреляционной и регрессионной матриц.
        """
        try:
            if not AutoTest.results:
                return
            from sklearn.linear_model import LinearRegression
            df = pd.DataFrame(AutoTest.results)
            params = ['n_heads', 'n_layers', 'd_model']
            metrics = ['train_loss', 'val_loss', 'bleu']
            df = df[df['train_loss'] != float('inf')][df['val_loss'] != float('inf')]
            if df.empty:
                self.log_message("Недостаточно данных для построения матриц", Fore.YELLOW)
                return
            corr_matrix = df[params + metrics].corr(method='pearson').round(4)
            self.correlation_text.config(state='normal')
            self.correlation_text.delete(1.0, tk.END)
            self.correlation_text.insert(tk.END, "Корреляционная матрица (Пирсон):\n")
            self.correlation_text.insert(tk.END, str(corr_matrix))
            self.correlation_text.config(state='disabled')
            regression_results = {}
            for metric in metrics:
                X = df[params].values
                y = df[metric].values
                reg = LinearRegression().fit(X, y)
                regression_results[metric] = reg.coef_
            reg_matrix = pd.DataFrame(regression_results, index=params).round(4)
            self.regression_text.config(state='normal')
            self.regression_text.delete(1.0, tk.END)
            self.regression_text.insert(tk.END, "Регрессионная матрица (коэффициенты):\n")
            self.regression_text.insert(tk.END, str(reg_matrix))
            self.regression_text.config(state='disabled')
            self.log_message("Корреляционная и регрессионная матрицы обновлены")
        except Exception as e:
            self.log_message(f"Ошибка обновления матриц: {str(e)}", Fore.RED)

    def update_training_progress(self, progress, epoch, total_epochs, step, steps_per_epoch, global_step, total_steps):
        """
        Обновление прогресса обучения.
        """
        try:
            self.log_message(f"Обновление прогресса: {progress:.1f}%, Эпоха: {epoch}/{total_epochs}, Шаг: {step}/{steps_per_epoch}, Всего шагов: {global_step}/{total_steps}", Fore.BLUE)
            self.progress.set(progress)
            self.progress_label.config(text=f"Прогресс: {progress:.1f}%")
            self.training_info.set(f"Эпоха: {epoch}/{total_epochs}, Шаг: {step}/{steps_per_epoch}, Всего шагов: {global_step}/{total_steps}")
            self.root.update()  # Принудительное обновление интерфейса
        except Exception as e:
            self.log_message(f"Ошибка обновления прогресса: {str(e)}", Fore.RED)

    def stop_training(self):
        """
        Остановка обучения.
        """
        try:
            if hasattr(self.trainer, 'stop_training'):
                self.trainer.stop_training()
                self.training_info.set("Эпоха: 0/0, Шаг: 0/0, Всего шагов: 0/0")
                self.progress.set(0)
                self.progress_label.config(text="Прогресс: 0%")
                self.log_message("Обучение остановлено", Fore.YELLOW)
            else:
                self.log_message("Ошибка: Метод stop_training не найден в объекте trainer", Fore.RED)
        except Exception as e:
            self.log_message(f"Ошибка остановки обучения: {str(e)}", Fore.RED)

    def translate(self):
        """
        Перевод введенного текста.
        """
        try:
            if not os.path.exists(self.config.model_path):
                messagebox.showerror("Ошибка", "Модель не найдена. Сначала обучите модель!")
                self.log_message("Ошибка: Модель не найдена", Fore.RED)
                return
            text = self.text_entry.get().strip()
            if not text:
                messagebox.showwarning("Предупреждение", "Введите текст для перевода!")
                self.log_message("Предупреждение: Текст для перевода не введен", Fore.YELLOW)
                return
            translation = self.trainer.translate(text)
            self.translation_result.config(text=f"Перевод на русский: {translation}")
            self.log_message(f"Перевод: '{text}' -> '{translation}'")
        except Exception as e:
            self.log_message(f"Ошибка перевода: {str(e)}", Fore.RED)

    def log_message(self, message, color=Fore.WHITE):
        """
        Добавление сообщения в лог.
        """
        try:
            self.log_text.config(state='normal')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
            self.log_text.config(state='disabled')
            self.log_text.see(tk.END)
            print(f"===GUI.py===\n{color}{message}{Style.RESET_ALL}")
        except Exception as e:
            print(f"===GUI.py===\n{Fore.RED}Ошибка логирования: {str(e)}{Style.RESET_ALL}")

    def clear_logs(self):
        """
        Очистка логов.
        """
        try:
            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state='disabled')
            self.log_message("Логи очищены", Fore.YELLOW)
        except Exception as e:
            self.log_message(f"Ошибка очистки логов: {str(e)}", Fore.RED)

    def configure_styles(self):
        """
        Настройка стилей для элементов интерфейса.
        """
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TCheckbutton", font=("Arial", 10))
        style.configure("green.Horizontal.TProgressbar", troughcolor='lightgray', background='green')

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root, None, None, None)
    gui.configure_styles()
    root.mainloop()