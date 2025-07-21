import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import os
import sacrebleu
from colorama import init, Fore, Style

init()

class Trainer:
    def __init__(self, model, data_processor, config):
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
        print(f"===trainer.py===\n{color}{message}{Style.RESET_ALL}")

    def train_tokenizer(self, run_id):
        self.log(f"Начало тренировки токенизатора для run_{run_id}...", Fore.YELLOW)
        try:
            special_tokens = self.config.special_tokens
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
        try:
            checkpoint_path = f"runs/run_{run_id}/checkpoint_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log(f"Чекпоинт сохранен в {checkpoint_path}", Fore.GREEN)
        except Exception as e:
            self.log(f"Ошибка сохранения чекпоинта: {str(e)}", Fore.RED)

    def load_checkpoint(self, run_id, epoch=None):
        try:
            checkpoint_dir = f"runs/run_{run_id}"
            if not os.path.exists(checkpoint_dir):
                self.log("Чекпоинты не найдены", Fore.YELLOW)
                return 0
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
            if not checkpoints:
                self.log("Чекпоинты не найдены", Fore.YELLOW)
                return 0
            if epoch is None:
                epoch = max(int(f.split("_epoch_")[1].split(".pt")[0]) for f in checkpoints)
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.log(f"Загружен чекпоинт {checkpoint_path}", Fore.GREEN)
            return epoch
        except Exception as e:
            self.log(f"Ошибка загрузки чекпоинта: {str(e)}", Fore.RED)
            return 0

    def log_architecture_params(self, run_id):
        """Логирование параметров архитектуры в TensorBoard."""
        # Словари для преобразования строковых параметров в числовые индексы
        attention_type_map = {
            "scaled_dot_product": 0,
            "multi_head": 1,
            "additive": 2
        }
        normalization_type_map = {
            "layer_norm": 0,
            "batch_norm": 1,
            "none": 2
        }
        layer_type_map = {
            "transformer": 0,
            "feed_forward": 1,
            "convolutional": 2
        }

        architecture_params = {
            'dropout': self.config.dropout,
            'dropout_attn': self.config.dropout_attn,
            'attention_type': attention_type_map.get(self.config.attention_type, -1),
            'normalization_type': normalization_type_map.get(self.config.normalization_type, -1),
            'layer_type': layer_type_map.get(self.config.layer_type, -1),
            'ffn_dim': self.config.ffn_dim,
            'norm_eps': self.config.norm_eps,
            'use_learnable_dropout': int(self.config.use_learnable_dropout),
            'use_learnable_dropout_attn': int(self.config.use_learnable_dropout_attn),
            'apply_residual': int(self.config.apply_residual),
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
        }
        for param, value in architecture_params.items():
            self.writer.add_scalar(f"Architecture/{param}", value, 0)
        # Дополнительно логируем строковые значения как текст для читаемости
        self.writer.add_text("Architecture/attention_type_str", self.config.attention_type, 0)
        self.writer.add_text("Architecture/normalization_type_str", self.config.normalization_type, 0)
        self.writer.add_text("Architecture/layer_type_str", self.config.layer_type, 0)

    def log_trainable_params(self, global_step):
        """Логирование обучаемых параметров в TensorBoard."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Parameters/{name}", param, global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

    def process_batch(self, src, tgt, optimizer, criterion):
        src, tgt = src.to(self.device), tgt.to(self.device)
        optimizer.zero_grad()
        output = self.model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, progress_callback=None, metric_queue=None, run_id=None):
        self.is_training = True
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/run_{run_id}")
        try:
            # Логируем параметры архитектуры в начале обучения
            self.log_architecture_params(run_id)
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
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            self.total_steps = len(train_dataloader) * self.config.epochs

            start_epoch = self.load_checkpoint(run_id)
            self.model.train()
            global_step = start_epoch * len(train_dataloader)
            log_interval = max(1, len(train_dataloader) // 10)

            for epoch in range(start_epoch, self.config.epochs):
                self.current_epoch = epoch + 1
                epoch_train_loss = 0.0
                progress_bar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), desc=f"Эпоха {self.current_epoch}/{self.config.epochs}")

                for step, (src, tgt) in progress_bar:
                    if not self.is_training:
                        self.log(f"Обучение прервано на эпохе {self.current_epoch}, шаге {step}", Fore.RED)
                        return
                    self.current_step = step
                    try:
                        loss = self.process_batch(src, tgt, optimizer, criterion)
                    except Exception as e:
                        self.log(f"Ошибка обработки батча на шаге {step}: {str(e)}", Fore.RED)
                        raise
                    epoch_train_loss += loss
                    self.writer.add_scalar("Loss/train", loss, global_step)
                    # Логируем обучаемые параметры каждые log_interval шагов
                    if step % log_interval == 0:
                        self.log_trainable_params(global_step)

                    if step % log_interval == 0:
                        self.log(f"Эпоха {self.current_epoch}/{self.config.epochs}, Шаг {step}/{len(train_dataloader)}, Loss: {loss:.4f}", Fore.BLUE)

                    if progress_callback:
                        progress = (global_step + 1) / self.total_steps * 100
                        progress_callback(progress, self.current_epoch, self.config.epochs, step, len(train_dataloader), global_step, self.total_steps)
                    global_step += 1

                epoch_train_loss /= len(train_dataloader)
                self.save_checkpoint(self.current_epoch, run_id)

                val_loss, bleu_score = self.validate(val_dataloader, criterion)
                self.writer.add_scalar("Loss/val", val_loss, self.current_epoch)
                self.writer.add_scalar("BLEU/val", bleu_score, self.current_epoch)

                if metric_queue:
                    metric_queue.put({
                        'train_loss': epoch_train_loss,
                        'val_loss': val_loss,
                        'bleu': bleu_score
                    })
                self.log(f"Эпоха {self.current_epoch} Метрики: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}", Fore.GREEN)

            torch.save(self.model.state_dict(), self.config.model_path)
            self.log(f"Обучение завершено, модель сохранена в {self.config.model_path}", Fore.GREEN)
        except Exception as e:
            self.log(f"Ошибка обучения: {str(e)}", Fore.RED)
            raise
        finally:
            self.writer.close()
            self.is_training = False

    def validate(self, dataloader, criterion):
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        references = []

        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                output = self.model(src, tgt[:, :-1])
                loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
                val_steps += 1

                pred_tokens = output.argmax(dim=-1).cpu().numpy()
                for pred, ref in zip(pred_tokens, tgt.cpu().numpy()):
                    pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
                    ref_text = self.tokenizer.decode(ref[1:], skip_special_tokens=True)
                    if ref_text.strip():
                        predictions.append(pred_text)
                        references.append([ref_text])

        val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        bleu_score = sacrebleu.corpus_bleu(predictions, references).score if predictions else 0.0
        self.model.train()
        return val_loss, bleu_score

    def stop_training(self):
        self.is_training = False
        self.log("Запрос на остановку обучения", Fore.RED)

    def translate(self, text):
        self.log(f"Начало перевода текста: {text}", Fore.YELLOW)
        try:
            self.model.eval()
            tokens = self.tokenizer.encode(text).ids
            src = torch.tensor([tokens[:self.config.max_len]], dtype=torch.long).to(self.device)
            src = torch.nn.functional.pad(src, (0, self.config.max_len - src.size(1)), value=self.tokenizer.token_to_id("[PAD]"))
            tgt = torch.tensor([[self.tokenizer.token_to_id("[CLS]")]], dtype=torch.long).to(self.device)

            for _ in range(self.config.max_len - 1):
                output = self.model(src, tgt)
                next_token = output[:, -1, :].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=-1)
                if next_token.item() == self.tokenizer.token_to_id("[SEP]"):
                    break

            translation = self.tokenizer.decode(tgt[0].cpu().numpy(), skip_special_tokens=True)
            self.log(f"Результат перевода: {translation}", Fore.GREEN)
            return translation
        except Exception as e:
            self.log(f"Ошибка перевода: {str(e)}", Fore.RED)
            return ""