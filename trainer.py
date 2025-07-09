import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import os
import time
import sacrebleu
from colorama import init, Fore, Style

init()


class Trainer:
    def __init__(self, model, data_processor, config):
        """
        Инициализация тренера для обучения модели машинного перевода.

        Args:
            model: Модель для обучения.
            data_processor: Объект для обработки данных.
            config: Конфигурация с параметрами обучения.
        """
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = None
        self.writer = None
        self.is_training = False
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        os.makedirs(os.path.dirname(self.config.tokenizer_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.config.model_path) or '.', exist_ok=True)
        self.log(f"Тренер инициализирован с устройством: {self.device}", Fore.BLUE)

    def log(self, message, color=Fore.WHITE):
        """Унифицированное логирование сообщений."""
        print(f"===trainer.py===\n{color}{message}{Style.RESET_ALL}")

    def train_tokenizer(self, run_id):
        """
        Тренировка токенизатора (BPE или WordPiece) на основе конфигурации.

        Args:
            run_id: Идентификатор текущего запуска.

        Returns:
            Tokenizer: Обученный токенизатор.
        """
        self.log(f"Начало тренировки токенизатора для run_{run_id}...", Fore.YELLOW)
        try:
            special_tokens = self.config.special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            if self.config.tokenizer_type == "WordPiece":
                tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
                trainer = WordPieceTrainer(vocab_size=self.config.vocab_size, special_tokens=special_tokens)
            else:
                tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
                trainer = BpeTrainer(vocab_size=self.config.vocab_size, special_tokens=special_tokens)
            tokenizer.pre_tokenizer = Whitespace()

            if not self.config.dataset_paths:
                datasets_info = self.data_processor.check_datasets()
                self.config.dataset_paths = [
                    os.path.join(self.config.datasets_dir, f) for f in datasets_info.keys() if datasets_info[f]["valid"]
                ]
                self.log(f"Установлены пути к датасетам: {self.config.dataset_paths}", Fore.CYAN)

            tokenizer.train(self.config.dataset_paths, trainer)
            tokenizer_path = f"runs/run_{run_id}/tokenizer.json"
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            tokenizer.save(tokenizer_path)
            self.log(f"Токенизатор обучен и сохранен в {tokenizer_path}", Fore.GREEN)
            return tokenizer
        except Exception as e:
            self.log(f"Ошибка тренировки токенизатора: {str(e)}", Fore.RED)
            raise

    def save_checkpoint(self, epoch, run_id):
        """
        Сохранение промежуточной модели (чекпоинта).

        Args:
            epoch: Номер текущей эпохи.
            run_id: Идентификатор текущего запуска.
        """
        try:
            checkpoint_path = f"runs/run_{run_id}/checkpoint_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log(f"Чекпоинт сохранен в {checkpoint_path}", Fore.GREEN)
        except Exception as e:
            self.log(f"Ошибка сохранения чекпоинта: {str(e)}", Fore.RED)

    def load_checkpoint(self, run_id, epoch=None):
        """
        Загрузка чекпоинта модели.

        Args:
            run_id: Идентификатор запуска.
            epoch: Номер эпохи для загрузки (если None, загружается последний).

        Returns:
            int: Номер загруженной эпохи или 0, если чекпоинт не найден.
        """
        try:
            checkpoint_dir = f"runs/run_{run_id}"
            if not os.path.exists(checkpoint_dir):
                self.log("Чекпоинты не найдены", Fore.YELLOW)
                return 0
            checkpoints = [f for f in os.listdir(checkpoint_dir) if
                           f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
            if not checkpoints:
                self.log("Чекпоинты не найдены", Fore.YELLOW)
                return 0
            if epoch is None:
                epoch = max(int(f.split("_epoch_")[1].split(".pt")[0]) for f in checkpoints)
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.log(f"Загружен чекпоинт {checkpoint_path}", Fore.GREEN)
            return epoch
        except Exception as e:
            self.log(f"Ошибка загрузки чекпоинта: {str(e)}", Fore.RED)
            return 0

    def process_batch(self, src, tgt, optimizer, criterion):
        """
        Обработка одного батча во время обучения.

        Args:
            src: Исходные данные батча.
            tgt: Целевые данные батча.
            optimizer: Оптимизатор.
            criterion: Функция потерь.

        Returns:
            float: Значение потерь для батча.
        """
        src, tgt = src.to(self.device), tgt.to(self.device)
        optimizer.zero_grad()
        output = self.model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, progress_callback=None, metric_queue=None, run_id=None):
        """
        Основной метод обучения модели.

        Args:
            progress_callback: Функция для обновления прогресса в GUI.
            metric_queue: Очередь для отправки метрик в GUI.
            run_id: Идентификатор текущего запуска.
        """
        self.is_training = True
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/run_{run_id}")
        try:
            self.tokenizer = self.train_tokenizer(run_id)
            self.config.tokenizer_path = f"runs/run_{run_id}/tokenizer.json"
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("[PAD]"))

            train_dataset, val_dataset = self.data_processor.load_datasets()
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size,
                                                           shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            self.total_steps = len(train_dataloader) * self.config.epochs

            start_epoch = self.load_checkpoint(run_id)
            self.model.train()
            global_step = start_epoch * len(train_dataloader)
            log_interval = max(1, len(train_dataloader) // 10)  # Логировать 10 раз за эпоху

            for epoch in range(start_epoch, self.config.epochs):
                self.current_epoch = epoch + 1
                epoch_train_loss = 0.0
                progress_bar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader),
                                    desc=f"Эпоха {self.current_epoch}/{self.config.epochs}")

                for step, (src, tgt) in progress_bar:
                    if not self.is_training:
                        self.log(f"Обучение прервано на эпохе {self.current_epoch}, шаге {step}", Fore.RED)
                        return
                    self.current_step = step
                    loss = self.process_batch(src, tgt, optimizer, criterion)
                    epoch_train_loss += loss
                    self.writer.add_scalar("Loss/train", loss, global_step)

                    if step % log_interval == 0:
                        self.log(
                            f"Эпоха {self.current_epoch}/{self.config.epochs}, Шаг {step}/{len(train_dataloader)}, Loss: {loss:.4f}, Global Step: {global_step}",
                            Fore.BLUE)

                    if progress_callback:
                        progress = (global_step + 1) / self.total_steps * 100
                        progress_callback(progress, self.current_epoch, self.config.epochs, step, len(train_dataloader),
                                          global_step, self.total_steps)
                    global_step += 1

                epoch_train_loss /= len(train_dataloader)
                self.save_checkpoint(self.current_epoch, run_id)

                # Валидация
                val_loss, bleu_score = self.validate(val_dataloader, criterion)
                self.writer.add_scalar("Loss/val", val_loss, self.current_epoch)
                self.writer.add_scalar("BLEU/val", bleu_score, self.current_epoch)

                if metric_queue:
                    metric_queue.put({
                        'train_loss': epoch_train_loss,
                        'val_loss': val_loss,
                        'bleu': bleu_score
                    })
                self.log(
                    f"Эпоха {self.current_epoch} Метрики: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}",
                    Fore.GREEN)

            torch.save(self.model.state_dict(), self.config.model_path)
            self.log(f"Обучение завершено, модель сохранена в {self.config.model_path}", Fore.GREEN)
        except Exception as e:
            self.log(f"Ошибка обучения: {str(e)}", Fore.RED)
            raise
        finally:
            self.writer.close()
            self.is_training = False

    def validate(self, dataloader, criterion):
        """
        Валидация модели на валидационной выборке.

        Args:
            dataloader: DataLoader для валидационных данных.
            criterion: Функция потерь.

        Returns:
            tuple: Средняя потеря на валидации и BLEU-метрика.
        """
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        references = []

        try:
            with torch.no_grad():
                for src, tgt in dataloader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    output = self.model(src, tgt[:, :-1])
                    loss = criterion(output.contiguous().view(-1, self.config.vocab_size),
                                     tgt[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
                    val_steps += 1

                    # Пакетное вычисление BLEU
                    pred_tokens = output.argmax(dim=-1).cpu().numpy()
                    for pred, ref in zip(pred_tokens, tgt.cpu().numpy()):
                        pred_text = self.tokenizer.decode(pred)
                        ref_text = self.tokenizer.decode(ref[1:])  # Пропуск [CLS]
                        if ref_text.strip():
                            predictions.append(pred_text)
                            references.append([ref_text])

            val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            bleu_score = sacrebleu.corpus_bleu(predictions, references).score if predictions else 0.0
            return val_loss, bleu_score
        except Exception as e:
            self.log(f"Ошибка валидации: {str(e)}", Fore.RED)
            return float('inf'), 0.0
        finally:
            self.model.train()

    def stop_training(self):
        """Прерывание процесса обучения."""
        self.is_training = False
        self.log("Запрос на остановку обучения", Fore.RED)

    def translate(self, text):
        """
        Перевод текста с использованием обученной модели.

        Args:
            text: Входной текст для перевода.

        Returns:
            str: Переведенный текст.
        """
        self.log(f"Начало перевода текста: {text}", Fore.YELLOW)
        try:
            self.model.eval()
            tokens = self.tokenizer.encode(text).ids
            src = torch.tensor([tokens], dtype=torch.long).to(self.device)
            tgt = torch.tensor([[self.tokenizer.token_to_id("[CLS]")]], dtype=torch.long).to(self.device)

            for i in range(self.config.max_len):
                output = self.model(src, tgt)
                next_token = output[:, -1, :].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=-1)
                if next_token.item() == self.tokenizer.token_to_id("[SEP]"):
                    self.log(f"Перевод завершен на токене {i + 1}", Fore.GREEN)
                    break

            translation = self.tokenizer.decode(tgt[0].cpu().numpy())
            self.log(f"Результат перевода: {translation}", Fore.GREEN)
            return translation
        except Exception as e:
            self.log(f"Ошибка перевода: {str(e)}", Fore.RED)
            return ""