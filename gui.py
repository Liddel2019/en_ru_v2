import os
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import time
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
        self.training_info = tk.StringVar(value="Epoch: 0/0, Step: 0/0, Total Steps: 0/0")
        self.runs = {}  # Dictionary to store metrics for each run: {run_id: {'train_loss': [], 'val_loss': [], 'bleu': []}}
        self.current_run_id = None
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        print(f"===GUI.py===\n{Fore.BLUE}Initializing GUI...{Style.RESET_ALL}")

        # Настройка окна
        self.root.resizable(True, True)
        self.root.minsize(600, 400)

        # Создание вкладок
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Создание фреймов для вкладок
        self.training_frame = ttk.Frame(self.notebook)
        self.datasets_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.translation_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.datasets_frame, text="Datasets")
        self.notebook.add(self.metrics_frame, text="Metrics")
        self.notebook.add(self.translation_frame, text="Translation")

        # Инициализация вкладок
        self.create_training_tab()
        self.create_datasets_tab()
        self.create_metrics_tab()
        self.create_translation_tab()

        # Запуск проверки очередей
        self.check_download_queue()
        self.check_metric_queue()
        print(f"===GUI.py===\n{Fore.GREEN}GUI initialized successfully{Style.RESET_ALL}")

    def create_training_tab(self):
        """Создание вкладки обучения"""
        self.training_info_label = ttk.Label(self.training_frame, textvariable=self.training_info)
        self.training_info_label.pack(padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(self.training_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(padx=10, pady=5, fill="x")
        self.progress_label = ttk.Label(self.training_frame, text="Progress: 0%")
        self.progress_label.pack(padx=10, pady=5)

        # Настройки
        self.settings_frame = ttk.LabelFrame(self.training_frame, text="Settings")
        self.settings_frame.pack(padx=10, pady=10, fill="x")
        self.create_settings_fields()

        # Выбор датасетов
        self.dataset_frame = ttk.LabelFrame(self.training_frame, text="Select Datasets")
        self.dataset_frame.pack(padx=10, pady=10, fill="x")
        self.dataset_list = {}

        # Фильтры по длине предложений
        self.filter_frame = ttk.LabelFrame(self.training_frame, text="Sentence Length Filters")
        self.filter_frame.pack(padx=10, pady=10, fill="x")
        self.create_filter_fields()

        ttk.Button(self.training_frame, text="Refresh Dataset List", command=self.update_dataset_list).pack(pady=5)
        ttk.Button(self.training_frame, text="Start Training", command=self.start_training).pack(pady=5)
        ttk.Button(self.training_frame, text="Stop Training", command=self.stop_training).pack(pady=5)

    def create_datasets_tab(self):
        """Создание вкладки датасетов"""
        self.available_datasets_frame = ttk.LabelFrame(self.datasets_frame, text="Available Datasets")
        self.available_datasets_frame.pack(padx=10, pady=10, fill="x")

        self.download_frame = ttk.LabelFrame(self.datasets_frame, text="Datasets for Download")
        self.download_frame.pack(padx=10, pady=10, fill="x")
        self.download_list = {}

    def create_metrics_tab(self):
        """Создание вкладки метрик"""
        # Dropdown for metric type
        metric_options = ["Training Loss", "Validation Loss", "BLEU Score"]
        self.selected_metric = tk.StringVar(value=metric_options[0])
        self.metric_dropdown = ttk.OptionMenu(self.metrics_frame, self.selected_metric, *metric_options, command=lambda _: self.update_plot())
        self.metric_dropdown.pack(pady=5)

        # Test run selection
        self.test_selection_frame = ttk.LabelFrame(self.metrics_frame, text="Select Test Runs")
        self.test_selection_frame.pack(padx=10, pady=5, fill="x")
        self.test_vars = {}
        self.update_test_selection()

        # Metric labels
        self.train_loss_label = ttk.Label(self.metrics_frame, text="Training Loss: N/A")
        self.val_loss_label = ttk.Label(self.metrics_frame, text="Validation Loss: N/A")
        self.bleu_label = ttk.Label(self.metrics_frame, text="BLEU Score: N/A")
        self.train_loss_label.pack(pady=2)
        self.val_loss_label.pack(pady=2)
        self.bleu_label.pack(pady=2)

        # Plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.metrics_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_translation_tab(self):
        """Создание вкладки перевода"""
        self.text_entry = tk.Entry(self.translation_frame, width=50)
        self.text_entry.pack(padx=5, pady=5, fill="x")

        ttk.Button(self.translation_frame, text="Translate", command=self.translate).pack(pady=5)

        self.translation_result = ttk.Label(self.translation_frame, text="")
        self.translation_result.pack(padx=5, pady=5, fill="x")

    def create_settings_fields(self):
        """Создание полей настроек с использованием Spinbox"""
        settings = [
            ("batch_size", "Batch Size", 1, 100, 1, 0),
            ("epochs", "Epochs", 1, 50, 1, 0),
            ("learning_rate", "Learning Rate", 0.00001, 1.0, 0.00001, 5),
            ("max_len", "Max Length", 1, 100, 1, 0),
            ("d_model", "Model Size", 32, 512, 8, 0),
            ("n_heads", "Heads", 1, 16, 1, 0),
            ("n_layers", "Layers", 1, 12, 1, 0),
            ("dropout", "Dropout", 0.0, 0.5, 0.01, 2),
            ("val_split", "Validation Split", 0.0, 1.0, 0.01, 2),
            ("min_chars", "Min Sentence Length", 1, 100, 1, 0),
            ("max_chars", "Max Sentence Length", 1, 500, 10, 0),
        ]
        self.settings_vars = {}
        for i, (param, label, min_val, max_val, increment, decimals) in enumerate(settings):
            frame = ttk.Frame(self.settings_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value=str(getattr(self.config, param)))
            self.settings_vars[param] = var
            spinbox = ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, increment=increment,
                                  width=10, format=f"%.{decimals}f" if decimals > 0 else "%d")
            spinbox.grid(row=0, column=1, sticky="e")
            spinbox.config(validate="key", validatecommand=(self.root.register(self.validate_input), '%P', str(min_val), str(max_val), param))
        self.settings_frame.columnconfigure(0, weight=1)

    def create_filter_fields(self):
        """Создание полей для фильтрации по длине предложений"""
        self.length_categories = {
            "Short": (1, 50),
            "Medium": (51, 100),
            "Long": (101, 200)
        }
        self.filter_vars = {}
        for i, (category, (min_len, max_len)) in enumerate(self.length_categories.items()):
            frame = ttk.Frame(self.filter_frame)
            frame.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            var = tk.BooleanVar(value=True)
            self.filter_vars[category] = var
            ttk.Checkbutton(frame, text=f"{category} Sentences ({min_len}-{max_len} chars)", variable=var).grid(row=0, column=0, sticky="w")
        self.filter_frame.columnconfigure(0, weight=1)

    def validate_input(self, value, min_val, max_val, param):
        """Валидация ввода с улучшенной обработкой ошибок"""
        if not value or value == '-':
            return True
        try:
            # Удаляем пробелы и проверяем формат
            value = value.strip()
            if value.startswith('-') and len(value) > 1:
                return False
            val = float(value) if param in ['learning_rate', 'dropout', 'val_split'] else int(value)
            min_val = float(min_val) if '.' in min_val else int(min_val)
            max_val = float(max_val) if '.' in max_val else int(max_val)
            if min_val <= val <= max_val:
                setattr(self.config, param, val)
                print(f"===GUI.py===\n{Fore.YELLOW}Updated config: {param}={val}{Style.RESET_ALL}")
                return True
            return False
        except ValueError:
            print(f"===GUI.py===\n{Fore.RED}Invalid input for {param}: {value}{Style.RESET_ALL}")
            return False

    def update_dataset_list(self):
        """Обновление списка датасетов с учетом фильтров"""
        datasets_info = self.data_processor.check_datasets()
        for widget in self.available_datasets_frame.winfo_children():
            widget.destroy()
        for name, info in datasets_info.items():
            var = tk.BooleanVar(value=info["valid"])
            chk = ttk.Checkbutton(self.available_datasets_frame, text=f"{name} ({info['num_rows']} rows, {info['reason']})",
                                 variable=var)
            chk.pack(side="top", fill="x")
            self.dataset_list[name] = {"var": var, "info": info}
            print(f"===GUI.py===\n{Fore.CYAN}Added dataset: {name}{Style.RESET_ALL}")

        for widget in self.download_frame.winfo_children():
            widget.destroy()
        for name, info in datasets_info.items():
            if "url" in info:
                btn = ttk.Button(self.download_frame, text=f"Download {name}",
                                command=lambda n=name, u=info["url"]: self.start_download(n, u))
                btn.pack(side="top", fill="x")
                self.download_list[name] = btn

    def start_download(self, dataset_name, url):
        """Запуск загрузки датасета"""
        def download_thread():
            def update_progress(block_num, block_size, total_size):
                if total_size > 0:
                    progress = (block_num * block_size) / total_size * 100
                    self.progress.set(progress)
                    self.progress_label.config(text=f"Progress: {progress:.1f}%")
            success, message = self.trainer.data_processor.download_dataset(dataset_name, url, update_progress)
            self.download_queue.put((dataset_name, success, message))

        threading.Thread(target=download_thread, daemon=True).start()

    def check_download_queue(self):
        """Проверка очереди загрузок"""
        try:
            while True:
                dataset_name, success, message = self.download_queue.get_nowait()
                if success:
                    messagebox.showinfo("Success", f"{dataset_name} downloaded: {message}")
                else:
                    messagebox.showerror("Error", f"Failed to download {dataset_name}: {message}")
                self.update_dataset_list()
        except queue.Empty:
            pass
        self.root.after(100, self.check_download_queue)

    def check_metric_queue(self):
        """Проверка очереди метрик"""
        try:
            while True:
                metrics = self.metric_queue.get_nowait()
                if self.current_run_id not in self.runs:
                    self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': []}
                self.runs[self.current_run_id]['train_loss'].append(metrics['train_loss'])
                self.runs[self.current_run_id]['val_loss'].append(metrics['val_loss'])
                self.runs[self.current_run_id]['bleu'].append(metrics['bleu'])
                self.train_loss_label.config(text=f"Training Loss: {metrics['train_loss']:.4f}")
                self.val_loss_label.config(text=f"Validation Loss: {metrics['val_loss']:.4f}")
                self.bleu_label.config(text=f"BLEU Score: {metrics['bleu']:.4f}")
                self.update_test_selection()
                self.update_plot()
        except queue.Empty:
            pass
        self.root.after(100, self.check_metric_queue)

    def update_test_selection(self):
        """Обновление списка тестов для сравнения"""
        for widget in self.test_selection_frame.winfo_children():
            widget.destroy()
        for run_id in self.runs:
            var = tk.BooleanVar(value=False)
            self.test_vars[run_id] = var
            ttk.Checkbutton(self.test_selection_frame, text=f"Run {run_id}",
                           variable=var, command=self.update_plot).pack(side="top", fill="x")

    def update_plot(self):
        """Обновление графика с учетом выбранных тестов"""
        self.ax.clear()
        metric = self.selected_metric.get()
        metric_key = {'Training Loss': 'train_loss', 'Validation Loss': 'val_loss', 'BLEU Score': 'bleu'}[metric]
        for idx, (run_id, run_data) in enumerate(self.runs.items()):
            if self.test_vars.get(run_id, tk.BooleanVar(value=False)).get() and run_data[metric_key]:
                self.ax.plot(range(1, len(run_data[metric_key]) + 1), run_data[metric_key], label=f"Run {run_id}", color=self.colors[idx % len(self.colors)])
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(metric)
        self.ax.set_title(f"{metric} over Epochs")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def get_selected_filters(self):
        """Получение выбранных фильтров длины предложений"""
        selected_filters = {category: (min_len, max_len) for category, (min_len, max_len) in self.length_categories.items() if self.filter_vars[category].get()}
        return selected_filters

    def start_training(self):
        """Запуск обучения"""
        selected_datasets = [name for name, info in self.dataset_list.items() if info["var"].get() and info["info"]["valid"]]
        if not selected_datasets:
            messagebox.showwarning("Warning", "Please select at least one valid dataset!")
            return
        self.config.dataset_paths = [os.path.join(self.config.datasets_dir, name) for name in selected_datasets]
        self.data_processor.length_filters = self.get_selected_filters()
        if not self.data_processor.length_filters:
            messagebox.showwarning("Warning", "Please select at least one sentence length filter!")
            return

        # Reset metrics for new run
        self.current_run_id = int(time.time())
        self.runs[self.current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': []}
        self.train_loss_label.config(text="Training Loss: N/A")
        self.val_loss_label.config(text="Validation Loss: N/A")
        self.bleu_label.config(text="BLEU Score: N/A")

        # Update model path for this run (tokenizer path is set in trainer.py)
        run_dir = f"runs/run_{self.current_run_id}"
        os.makedirs(run_dir, exist_ok=True)
        self.config.model_path = f"{run_dir}/model.pt"

        # Save configuration
        with open(f"{run_dir}/config.txt", 'w') as f:
            for attr, value in vars(self.config).items():
                f.write(f"{attr}: {value}\n")
        print(f"===GUI.py===\n{Fore.GREEN}Configuration saved to {run_dir}/config.txt{Style.RESET_ALL}")

        self.trainer.model = self.trainer.model.__class__(self.config)
        self.trainer.model.to(self.trainer.device)

        def training_thread():
            self.trainer.train(self.update_training_progress, self.metric_queue, self.current_run_id)
            # Save metrics as CSV and plot as PNG
            if self.current_run_id in self.runs:
                metrics_df = pd.DataFrame(self.runs[self.current_run_id])
                metrics_df.to_csv(f"{run_dir}/metrics.csv", index_label='epoch')
                print(f"===GUI.py===\n{Fore.GREEN}Metrics saved to {run_dir}/metrics.csv{Style.RESET_ALL}")

                # Save plots
                for metric, key in [('Training Loss', 'train_loss'), ('Validation Loss', 'val_loss'), ('BLEU Score', 'bleu')]:
                    plt.figure()
                    plt.plot(range(1, len(self.runs[self.current_run_id][key]) + 1), self.runs[self.current_run_id][key], label=metric, color=self.colors[0])
                    plt.xlabel("Epoch")
                    plt.ylabel(metric)
                    plt.title(f"{metric} for Run {self.current_run_id}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{run_dir}/{key}.png")
                    plt.close()
                print(f"===GUI.py===\n{Fore.GREEN}Plots saved to {run_dir}{Style.RESET_ALL}")

        threading.Thread(target=training_thread, daemon=True).start()

    def update_training_progress(self, progress, epoch, total_epochs, step, steps_per_epoch, global_step, total_steps):
        """Обновление прогресса обучения"""
        self.progress.set(progress)
        self.progress_label.config(text=f"Progress: {progress:.1f}%")
        self.training_info.set(f"Epoch: {epoch}/{total_epochs}, Step: {step}/{steps_per_epoch}, Total Steps: {global_step}/{total_steps}")

    def stop_training(self):
        """Остановка обучения"""
        if self.trainer:
            self.trainer.stop_training()
            self.training_info.set("Epoch: 0/0, Step: 0/0, Total Steps: 0/0")
            self.progress.set(0)
            self.progress_label.config(text="Progress: 0%")

    def translate(self):
        """Перевод текста"""
        if not os.path.exists(self.config.model_path):
            messagebox.showerror("Error", "Model not found. Train the model first!")
            return
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to translate!")
            return
        translation = self.trainer.translate(text)
        self.translation_result.config(text=f"Russian translation: {translation}")