import os
import datetime
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer
from tkinter import messagebox
from colorama import init, Fore, Style
import threading

init()

class AutoTest:
    results = []  # Статический список для хранения результатов автотестов

    def __init__(self, trainer, config, data_processor, runs, autotest_tree, log_message, update_matrices, metric_queue):
        """
        Инициализация автотеста с байесовской оптимизацией гиперпараметров.
        """
        self.trainer = trainer
        self.config = config
        self.data_processor = data_processor
        self.runs = runs
        self.autotest_tree = autotest_tree
        self.log_message = log_message
        self.update_matrices = update_matrices
        self.metric_queue = metric_queue

    def start(self, n_heads_min, n_heads_max, n_layers_min, n_layers_max, d_model_min, d_model_max, n_calls, optimize_metric, optimize_direction):
        """
        Запуск автотеста с байесовской оптимизацией.
        """
        space = [
            Integer(n_heads_min, n_heads_max, name='n_heads'),
            Integer(n_layers_min, n_layers_max, name='n_layers'),
            Integer(d_model_min, d_model_max, name='d_model')
        ]

        def objective(params):
            n_heads, n_layers, d_model = params
            if d_model % n_heads != 0:
                self.log_message(f"Пропущена конфигурация: d_model={d_model} не кратен n_heads={n_heads}", Fore.YELLOW)
                return 0.0 if optimize_metric == "BLEU Score" else float('inf')
            self.config.n_heads = n_heads
            self.config.n_layers = n_layers
            self.config.d_model = d_model

            current_run_id = int(datetime.datetime.now().timestamp())
            self.runs[current_run_id] = {'train_loss': [], 'val_loss': [], 'bleu': [], 'time': []}
            run_dir = f"runs/run_{current_run_id}"
            os.makedirs(run_dir, exist_ok=True)
            self.config.model_path = f"{run_dir}/model.pt"

            with open(f"{run_dir}/config.txt", 'w', encoding='utf-8') as f:
                for attr, value in vars(self.config).items():
                    f.write(f"{attr}: {value}\n")
            self.log_message(f"Конфигурация автотеста сохранена в {run_dir}/config.txt", Fore.GREEN)

            try:
                self.trainer.model = self.trainer.model.__class__(self.config)
                self.trainer.model.to(self.trainer.device)
                self.trainer.train(self.update_training_progress, self.metric_queue, current_run_id)
            except Exception as e:
                self.log_message(f"Ошибка обучения в автотесте: {str(e)}", Fore.RED)
                return 0.0 if optimize_metric == "BLEU Score" else float('inf')

            bleu_score = self.runs[current_run_id]['bleu'][-1] if self.runs[current_run_id]['bleu'] else 0.0
            train_loss = self.runs[current_run_id]['train_loss'][-1] if self.runs[current_run_id]['train_loss'] else float('inf')
            val_loss = self.runs[current_run_id]['val_loss'][-1] if self.runs[current_run_id]['val_loss'] else float('inf')

            AutoTest.results.append({
                'run_id': current_run_id,
                'n_heads': n_heads,
                'n_layers': n_layers,
                'd_model': d_model,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu': bleu_score
            })
            self.update_autotest_results()
            self.update_matrices()

            metric_value = {
                "BLEU Score": bleu_score,
                "Train Loss": train_loss,
                "Val Loss": val_loss
            }[optimize_metric]
            return -metric_value if optimize_direction == "Максимизировать" else metric_value

        def optimization_thread():
            try:
                self.log_message("Начало автотеста с байесовской оптимизацией")
                res = gp_minimize(
                    objective,
                    space,
                    n_calls=n_calls,
                    random_state=42,
                    verbose=True
                )
                best_metric = -res.fun if optimize_direction == "Максимизировать" else res.fun
                self.log_message(f"Автотест завершен. Лучший {optimize_metric}: {best_metric:.4f} при n_heads={res.x[0]}, n_layers={res.x[1]}, d_model={res.x[2]}", Fore.GREEN)
                messagebox.showinfo("Успех", f"Автотест завершен. Лучший {optimize_metric}: {best_metric:.4f} при n_heads={res.x[0]}, n_layers={res.x[1]}, d_model={res.x[2]}")
            except Exception as e:
                self.log_message(f"Ошибка автотеста: {str(e)}", Fore.RED)
                messagebox.showerror("Ошибка", f"Автотест завершился с ошибкой: {str(e)}")

        threading.Thread(target=optimization_thread, daemon=True).start()

    def update_training_progress(self, progress, epoch, total_epochs, step, steps_per_epoch, global_step, total_steps):
        """
        Обновление прогресса обучения (заглушка).
        """
        pass  # Реальная реализация в GUI

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