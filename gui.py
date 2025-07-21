import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
import threading
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from colorama import init, Fore, Style

init()

class GUI:
    def __init__(self, root, trainer, config, data_processor):
        self.root = root
        self.root.title("EN-RU Translator")
        self.config = config
        self.data_processor = data_processor
        self.trainer = trainer
        self.progress = tk.DoubleVar()
        self.download_queue = queue.Queue()
        self.metric_queue = queue.Queue()
        self.plot_queue = queue.Queue()
        self.training_info = tk.StringVar(value="Эпоха: 0/0, Шаг: 0/0, Всего шагов: 0/0")
        self.runs = {}
        self.current_run_id = None
        self.colors = ['blue', 'orange', 'green', 'red', 'purple']
        self.is_running = True
        print(f"===gui.py===\n{Fore.BLUE}Инициализация интерфейса...{Style.RESET_ALL}")

        self.root.resizable(True, True)
        self.root.minsize(800, 600)
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)

        self.training_frame = ttk.Frame(self.notebook)
        self.datasets_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.translation_frame = ttk.Frame(self.notebook)
        self.logs_frame = ttk.Frame(self.notebook)
        self.architecture_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.training_frame, text="Обучение")
        self.notebook.add(self.datasets_frame, text="Датасеты")
        self.notebook.add(self.metrics_frame, text="Метрики")
        self.notebook.add(self.translation_frame, text="Перевод")
        self.notebook.add(self.logs_frame, text="Логи")
        self.notebook.add(self.architecture_frame, text="Архитектура")

        self.create_training_tab()
        self.create_datasets_tab()
        self.create_metrics_tab()
        self.create_translation_tab()
        self.create_logs_tab()
        self.create_architecture_tab()

        self.update_dataset_list()
        self.check_download_queue()
        self.check_metric_queue()
        self.check_plot_queue()
        print(f"===gui.py===\n{Fore.GREEN}Интерфейс успешно инициализирован{Style.RESET_ALL}")

    def on_closing(self):
        self.is_running = False
        self.stop_training()
        self.root.destroy()

    def create_training_tab(self):
        self.training_info_label = ttk.Label(self.training_frame, textvariable=self.training_info, font=("Arial", 12))
        self.training_info_label.pack(padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(self.training_frame, variable=self.progress, maximum=100)
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
        ttk.Button(button_frame, text="Обновить список датасетов", command=self.update_dataset_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Начать обучение", command=self.start_training).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Остановить обучение", command=self.stop_training).pack(side="left", padx=5)

    def create_datasets_tab(self):
        self.available_datasets_frame = ttk.LabelFrame(self.datasets_frame, text="Доступные датасеты", padding="10")
        self.available_datasets_frame.pack(padx=10, pady=10, fill="x")

        self.download_frame = ttk.LabelFrame(self.datasets_frame, text="Датасеты для загрузки", padding="10")
        self.download_frame.pack(padx=10, pady=10, fill="x")
        self.download_list = {}

    def create_metrics_tab(self):
        metric_options = ["Потери на обучении", "Потери на валидации", "BLEU Score"]
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
        self.train_loss_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.val_loss_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.bleu_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.metrics_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_translation_tab(self):
        self.text_entry = ttk.Entry(self.translation_frame, width=50, font=("Arial", 12))
        self.text_entry.pack(padx=10, pady=10, fill="x")

        ttk.Button(self.translation_frame, text="Перевести", command=self.translate).pack(pady=5)

        self.translation_result = ttk.Label(self.translation_frame, text="", font=("Arial", 12), wraplength=600)
        self.translation_result.pack(padx=10, pady=10, fill="x")

    def create_logs_tab(self):
        self.log_text = scrolledtext.ScrolledText(self.logs_frame, height=20, font=("Arial", 10), state='disabled')
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)
        ttk.Button(self.logs_frame, text="Очистить логи", command=self.clear_logs).pack(pady=5)

    def create_architecture_tab(self):
        self.architecture_frame_inner = ttk.LabelFrame(self.architecture_frame, text="Параметры архитектуры", padding="10")
        self.architecture_frame_inner.pack(padx=10, pady=10, fill="x")

        architecture_settings = [
            ("attention_type", "Тип механизма внимания", ["scaled_dot_product", "multi_head", "additive"], "scaled_dot_product"),
            ("normalization_type", "Тип нормализации", ["layer_norm", "batch_norm", "none"], "layer_norm"),
            ("layer_type", "Тип слоев", ["transformer", "feed_forward", "convolutional"], "transformer"),
            ("activation", "Функция активации", ["relu", "gelu"], "gelu"),
            ("dropout", "Dropout", 0.0, 0.999, 0.01, 2),
            ("dropout_attn", "Dropout внимания", 0.0, 0.999, 0.01, 2),
            ("ffn_dim", "Размер FFN", 64, 2048, 64, 0),
            ("norm_eps", "Эпсилон нормализации", 1e-8, 1e-4, 1e-8, 8),
        ]
        self.architecture_vars = {}
        for i, setting in enumerate(architecture_settings):
            frame = ttk.Frame(self.architecture_frame_inner)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            if isinstance(setting[2], list):
                param, label, options, default = setting
                ttk.Label(frame, text=label, font=("Arial", 10)).grid(row=0, column=0, sticky="w")
                var = tk.StringVar(value=getattr(self.config, param, default))
                self.architecture_vars[param] = var
                ttk.OptionMenu(frame, var, default, *options, command=lambda _, p=param: self.update_architecture_param(p)).grid(row=0, column=1, sticky="e")
            else:
                param, label, min_val, max_val, increment, decimals = setting
                ttk.Label(frame, text=label, font=("Arial", 10)).grid(row=0, column=0, sticky="w")
                var = tk.StringVar(value=str(getattr(self.config, param)))
                self.architecture_vars[param] = var
                spinbox = ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, increment=increment,
                                      width=15, format=f"%.{decimals}f" if decimals > 0 else "%d", font=("Arial", 10))
                spinbox.grid(row=0, column=1, sticky="e")
                spinbox.bind("<KeyRelease>", lambda event, p=param: self.validate_spinbox_input(event, p))
                spinbox.bind("<<Increment>>", lambda event, p=param: self.validate_spinbox_input(event, p))
                spinbox.bind("<<Decrement>>", lambda event, p=param: self.validate_spinbox_input(event, p))

        self.use_learnable_dropout_var = tk.BooleanVar(value=self.config.use_learnable_dropout)
        ttk.Checkbutton(self.architecture_frame_inner, text="Использовать обучаемый dropout", variable=self.use_learnable_dropout_var,
                        command=lambda: self.update_architecture_param("use_learnable_dropout")).grid(row=len(architecture_settings), column=0, sticky="w", padx=5, pady=5)

        self.use_learnable_dropout_attn_var = tk.BooleanVar(value=self.config.use_learnable_dropout_attn)
        ttk.Checkbutton(self.architecture_frame_inner, text="Использовать обучаемый dropout внимания", variable=self.use_learnable_dropout_attn_var,
                        command=lambda: self.update_architecture_param("use_learnable_dropout_attn")).grid(row=len(architecture_settings)+1, column=0, sticky="w", padx=5, pady=5)

        self.apply_residual_var = tk.BooleanVar(value=self.config.apply_residual)
        ttk.Checkbutton(self.architecture_frame_inner, text="Использовать резидуальные соединения", variable=self.apply_residual_var,
                        command=lambda: self.update_architecture_param("apply_residual")).grid(row=len(architecture_settings)+2, column=0, sticky="w", padx=5, pady=5)

        self.architecture_frame_inner.columnconfigure(0, weight=1)

    def create_settings_fields(self):
        settings = [
            ("batch_size", "Размер пакета", 1, 1024, 1, 0),
            ("epochs", "Количество эпох", 1, 100, 1, 0),
            ("learning_rate", "Скорость обучения", 0.00001, 0.01, 0.00001, 5),
            ("max_len", "Максимальная длина", 8, 64, 1, 0),
            ("d_model", "Размер модели", 64, 512, 8, 0),
            ("n_heads", "Количество голов", 1, 16, 1, 0),
            ("n_layers", "Количество слоев", 1, 12, 1, 0),
            ("val_split", "Доля валидации", 0.1, 0.5, 0.01, 2),
            ("min_chars", "Мин. длина предложения", 1, 100, 1, 0),
            ("max_chars", "Макс. длина предложения", 10, 500, 10, 0),
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
            spinbox.bind("<KeyRelease>", lambda event, p=param: self.validate_spinbox_input(event, p))
            spinbox.bind("<<Increment>>", lambda event, p=param: self.validate_spinbox_input(event, p))
            spinbox.bind("<<Decrement>>", lambda event, p=param: self.validate_spinbox_input(event, p))
        self.settings_frame.columnconfigure(0, weight=1)

    def create_filter_fields(self):
        self.length_categories = {
            "Короткие": (1, 50),
            "Средние": (51, 100),
            "Длинные": (101, 200)
        }
        self.filter_vars = {}
        for i, (category, (min_len, max_len)) in enumerate(self.length_categories.items()):
            frame = ttk.Frame(self.filter_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            var = tk.BooleanVar(value=True)
            self.filter_vars[category] = var
            ttk.Checkbutton(frame, text=f"{category} предложения ({min_len}-{max_len} символов)", variable=var).grid(row=0, column=0, sticky="w")
        self.filter_frame.columnconfigure(0, weight=1)

    def validate_spinbox_input(self, event, param):
        value = event.widget.get()
        if not value or value == '-':
            return
        try:
            val = float(value) if param in ['learning_rate', 'dropout', 'val_split', 'dropout_attn', 'norm_eps'] else int(value)
            min_val = float(event.widget.cget("from")) if '.' in str(event.widget.cget("from")) else int(event.widget.cget("from"))
            max_val = float(event.widget.cget("to")) if '.' in str(event.widget.cget("to")) else int(event.widget.cget("to"))
            if min_val <= val <= max_val:
                setattr(self.config, param, val)
                self.log_message(f"Обновлен параметр: {param}={val}")
            else:
                self.log_message(f"Ошибка: Значение {param}={value} вне диапазона [{min_val}, {max_val}]", Fore.RED)
                event.widget.delete(0, tk.END)
                event.widget.insert(0, str(min_val))
        except ValueError:
            self.log_message(f"Ошибка: Неверный ввод для {param}: {value}", Fore.RED)
            event.widget.delete(0, tk.END)
            event.widget.insert(0, str(getattr(self.config, param)))

    def update_architecture_param(self, param):
        try:
            if param in ["use_learnable_dropout", "use_learnable_dropout_attn", "apply_residual"]:
                value = getattr(self, f"{param}_var").get()
            else:
                value = self.architecture_vars[param].get()
                if param in ["ffn_dim", "n_heads", "n_layers"]:
                    value = int(value)
                elif param in ["dropout", "dropout_attn", "norm_eps"]:
                    value = float(value)
            setattr(self.config, param, value)
            self.log_message(f"Обновлен параметр архитектуры: {param}={value}")
        except Exception as e:
            self.log_message(f"Ошибка обновления параметра архитектуры {param}: {str(e)}", Fore.RED)

    def update_dataset_list(self):
        try:
            datasets_info = self.data_processor.check_datasets()
            for widget in self.available_datasets_frame.winfo_children():
                widget.destroy()
            for name, info in datasets_info.items():
                var = tk.BooleanVar(value=info["valid"])
                chk = ttk.Checkbutton(self.available_datasets_frame, text=f"{name} ({info['num_rows']} строк)", variable=var)
                chk.pack(side="top", fill="x", pady=2)
                self.dataset_list[name] = {"var": var, "info": info}
                self.log_message(f"Добавлен датасет: {name}")

            for widget in self.download_frame.winfo_children():
                widget.destroy()
            for name, info in datasets_info.items():
                if "url" in info:
                    btn = ttk.Button(self.download_frame, text=f"Скачать {name}", command=lambda n=name, u=info["url"]: self.start_download(n, u))
                    btn.pack(side="top", fill="x", pady=2)
                    self.download_list[name] = btn
        except Exception as e:
            self.log_message(f"Ошибка при обновлении списка датасетов: {str(e)}", Fore.RED)

    def start_download(self, dataset_name, url):
        def download_thread():
            try:
                success, message = self.data_processor.download_dataset(dataset_name, url, self.update_download_progress)
                self.download_queue.put((dataset_name, success, message))
            except Exception as e:
                self.download_queue.put((dataset_name, False, f"Ошибка загрузки: {str(e)}"))

        self.log_message(f"Начало загрузки датасета: {dataset_name}")
        threading.Thread(target=download_thread, daemon=True).start()

    def update_download_progress(self, block_num, block_size, total_size):
        if total_size > 0:
            progress = (block_num * block_size) / total_size * 100
            self.progress.set(progress)
            self.progress_label.config(text=f"Прогресс загрузки: {progress:.1f}%")

    def check_download_queue(self):
        if not self.is_running:
            return
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
        if not self.is_running:
            return
        try:
            while True:
                metrics = self.metric_queue.get_nowait()
                if self.current_run_id not in self.runs:
                    self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': []}
                self.runs[self.current_run_id]['train_loss'].append(metrics['train_loss'])
                self.runs[self.current_run_id]['val_loss'].append(metrics['val_loss'])
                self.runs[self.current_run_id]['bleu'].append(metrics['bleu'])
                self.train_loss_label.config(text=f"Потери на обучении: {metrics['train_loss']:.4f}")
                self.val_loss_label.config(text=f"Потери на валидации: {metrics['val_loss']:.4f}")
                self.bleu_label.config(text=f"BLEU Score: {metrics['bleu']:.4f}")
                self.update_test_selection()
                self.update_plot()
                self.log_message(f"Обновлены метрики: Потери (обучение: {metrics['train_loss']:.4f}, валидация: {metrics['val_loss']:.4f}), BLEU: {metrics['bleu']:.4f}")
        except queue.Empty:
            pass
        self.root.after(100, self.check_metric_queue)

    def check_plot_queue(self):
        if not self.is_running:
            return
        try:
            while True:
                run_id, run_data = self.plot_queue.get_nowait()
                run_dir = f"runs/run_{run_id}"
                metrics_df = pd.DataFrame(run_data)
                metrics_df.to_csv(f"{run_dir}/metrics.csv", index_label='epoch')
                self.log_message(f"Метрики сохранены в {run_dir}/metrics.csv", Fore.GREEN)
        except queue.Empty:
            pass
        self.root.after(100, self.check_plot_queue)

    def update_test_selection(self):
        for widget in self.test_selection_frame.winfo_children():
            widget.destroy()
        for run_id in self.runs:
            var = tk.BooleanVar(value=False)
            self.test_vars[run_id] = var
            ttk.Checkbutton(self.test_selection_frame, text=f"Тест {run_id}", variable=var, command=self.update_plot).pack(side="top", fill="x", pady=2)

    def update_plot(self):
        self.ax.clear()
        metric = self.selected_metric.get()
        metric_key = {
            'Потери на обучении': 'train_loss',
            'Потери на валидации': 'val_loss',
            'BLEU Score': 'bleu'
        }[metric]
        has_data = False
        for idx, (run_id, run_data) in enumerate(self.runs.items()):
            if self.test_vars.get(run_id, tk.BooleanVar(value=False)).get() and run_data[metric_key]:
                self.ax.plot(range(1, len(run_data[metric_key]) + 1), run_data[metric_key], label=f"Тест {run_id}", color=self.colors[idx % len(self.colors)])
                has_data = True
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel(metric)
        self.ax.set_title(f"{metric} по эпохам")
        if has_data:
            self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def get_selected_filters(self):
        return {category: (min_len, max_len) for category, (min_len, max_len) in self.length_categories.items() if self.filter_vars[category].get()}

    def start_training(self):
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
        self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': []}
        self.train_loss_label.config(text="Потери на обучении: N/A")
        self.val_loss_label.config(text="Потери на валидации: N/A")
        self.bleu_label.config(text="BLEU Score: N/A")

        run_dir = f"runs/run_{self.current_run_id}"
        os.makedirs(run_dir, exist_ok=True)
        self.config.model_path = f"{run_dir}/model.pt"

        with open(f"{run_dir}/config.txt", 'w', encoding='utf-8') as f:
            for attr, value in vars(self.config).items():
                f.write(f"{attr}: {value}\n")
        self.log_message(f"Конфигурация сохранена в {run_dir}/config.txt", Fore.GREEN)

        try:
            self.config.validate()
            self.trainer.model = self.trainer.model.__class__(self.config)
            self.trainer.model.to(self.trainer.device)
        except Exception as e:
            self.log_message(f"Ошибка инициализации модели: {str(e)}", Fore.RED)
            return

        def training_thread():
            try:
                self.trainer.train(self.update_training_progress, self.metric_queue, self.current_run_id)
                if self.current_run_id in self.runs:
                    self.plot_queue.put((self.current_run_id, self.runs[self.current_run_id]))
            except Exception as e:
                self.log_message(f"Ошибка обучения: {str(e)}", Fore.RED)

        self.log_message("Начало обучения модели")
        threading.Thread(target=training_thread, daemon=True).start()

    def update_training_progress(self, progress, epoch, total_epochs, step, steps_per_epoch, global_step, total_steps):
        try:
            self.progress.set(progress)
            self.progress_label.config(text=f"Прогресс: {progress:.1f}%")
            self.training_info.set(f"Эпоха: {epoch}/{total_epochs}, Шаг: {step}/{steps_per_epoch}, Всего шагов: {global_step}/{total_steps}")
        except Exception as e:
            self.log_message(f"Ошибка обновления прогресса: {str(e)}", Fore.RED)

    def stop_training(self):
        try:
            self.trainer.stop_training()
            self.training_info.set("Эпоха: 0/0, Шаг: 0/0, Всего шагов: 0/0")
            self.progress.set(0)
            self.progress_label.config(text="Прогресс: 0%")
            self.log_message("Обучение остановлено", Fore.YELLOW)
        except Exception as e:
            self.log_message(f"Ошибка остановки обучения: {str(e)}", Fore.RED)

    def translate(self):
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
        try:
            self.log_text.config(state='normal')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
            self.log_text.config(state='disabled')
            self.log_text.see(tk.END)
            print(f"===gui.py===\n{color}{message}{Style.RESET_ALL}")
        except Exception as e:
            print(f"===gui.py===\n{Fore.RED}Ошибка логирования: {str(e)}{Style.RESET_ALL}")

    def clear_logs(self):
        try:
            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state='disabled')
            self.log_message("Логи очищены", Fore.YELLOW)
        except Exception as e:
            self.log_message(f"Ошибка очистки логов: {str(e)}", Fore.RED)