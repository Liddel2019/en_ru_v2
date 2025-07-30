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
import numpy as np
from collections import Counter

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
        self.dropout_params = []
        self.other_params = []
        for name, param in self.model.named_parameters():
            if 'dropout_rate' in name:
                self.dropout_params.append(param)
            else:
                self.other_params.append(param)
        if not self.other_params:
            self.log("Предупреждение: Список other_params пуст, проверьте инициализацию модели", Fore.YELLOW)
        if not self.dropout_params:
            self.log("Предупреждение: Список dropout_params пуст, обучаемый дропаут не используется", Fore.YELLOW)
        else:
            self.log(f"Обнаружено {len(self.dropout_params)} параметров dropout_rate", Fore.BLUE)
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
            checkpoints = [f for f in os.listdir(checkpoint_dir) if
                           f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
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
        architecture_params = {
            'dropout': self.config.dropout,
            'dropout_attn': self.config.dropout_attn,
            'dropout_lr': self.config.dropout_lr,
            'normalization_type': 0 if self.config.normalization_type == "layer_norm" else 1,
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
        self.writer.add_text("Architecture/normalization_type_str", self.config.normalization_type, 0)

    def log_trainable_params(self, global_step):
        if global_step % 10 != 0 and global_step % len(self.train_dataloader) != 0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Parameters/{name}", param, global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, global_step)
                if 'dropout_rate' in name:
                    self.writer.add_scalar(f"Dropout/{name}", param.item(), global_step)
                    if param.grad is not None:
                        self.writer.add_scalar(f"Gradients/Dropout/{name}", param.grad.item(), global_step)
                    else:
                        self.log(f"Предупреждение: Градиент для {name} отсутствует на шаге {global_step}", Fore.YELLOW)
                        self.writer.add_text(f"Dropout/{name}/warning", f"No gradient for {name} at step {global_step}",
                                             global_step)
        if global_step % len(self.train_dataloader) == 0:
            dropout_values = {name: param.item() for name, param in self.model.named_parameters() if
                              'dropout_rate' in name}
            self.log(f"Значения dropout_rate на шаге {global_step}: {dropout_values}", Fore.CYAN)

    def process_batch(self, src, tgt, optimizers, criterion):
        src, tgt = src.to(self.device), tgt.to(self.device)
        src_key_padding_mask = self.model.create_padding_mask(src, self.tokenizer.token_to_id("[PAD]"))
        tgt_key_padding_mask = self.model.create_padding_mask(tgt[:, :-1], self.tokenizer.token_to_id("[PAD]"))
        tgt_mask = self.model.create_look_ahead_mask(tgt.size(1) - 1)
        for optimizer in optimizers:
            optimizer.zero_grad()
        output = self.model(src, tgt[:, :-1], src_key_padding_mask, tgt_mask, tgt_key_padding_mask, writer=self.writer,
                            global_step=self.current_step)
        loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        for name, param in self.model.named_parameters():
            if 'dropout_rate' in name and param.grad is None:
                self.log(f"Предупреждение: Градиент для {name} отсутствует на шаге {self.current_step}", Fore.YELLOW)
                self.writer.add_text(f"Dropout/{name}/warning", f"No gradient for {name} at step {self.current_step}",
                                     self.current_step)
        for optimizer in optimizers:
            optimizer.step()
        return loss.item()

    def calculate_custom_metrics(self, predictions, references):
        word_count_match = []
        avg_word_len_match = []
        token_repetition = []
        for pred, refs in zip(predictions, references):
            pred_words = pred.split()
            ref_words = refs[0].split()

            # Совпадение количества слов
            word_count_match.append(1.0 if len(pred_words) == len(ref_words) else 0.0)

            # Среднее совпадение длины слова
            if pred_words and ref_words:
                pred_avg_len = np.mean([len(word) for word in pred_words])
                ref_avg_len = np.mean([len(word) for word in ref_words])
                len_diff = abs(pred_avg_len - ref_avg_len)
                avg_word_len_match.append(max(0.0, 1.0 - len_diff / max(pred_avg_len, ref_avg_len)))
            else:
                avg_word_len_match.append(0.0)

            # Частота повторения токенов
            pred_tokens = Counter(pred_words)
            max_repetition = max(pred_tokens.values()) if pred_tokens else 1
            token_repetition.append(1.0 / max_repetition if max_repetition > 1 else 0.0)

        return (
            np.mean(word_count_match) if word_count_match else 0.0,
            np.mean(avg_word_len_match) if avg_word_len_match else 0.0,
            np.mean(token_repetition) if token_repetition else 0.0
        )

    def calculate_rouge_metrics(self, predictions, references):
        def ngrams(words, n):
            return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

        rouge_1_scores = []
        rouge_l_scores = []
        for pred, refs in zip(predictions, references):
            pred_words = pred.split()
            ref_words = refs[0].split()

            # ROUGE-1
            pred_ngrams = set(ngrams(pred_words, 1))
            ref_ngrams = set(ngrams(ref_words, 1))
            common = len(pred_ngrams & ref_ngrams)
            rouge_1_precision = common / len(pred_ngrams) if pred_ngrams else 0.0
            rouge_1_recall = common / len(ref_ngrams) if ref_ngrams else 0.0
            rouge_1 = 2 * rouge_1_precision * rouge_1_recall / (rouge_1_precision + rouge_1_recall) if (
                                                                                                                   rouge_1_precision + rouge_1_recall) > 0 else 0.0
            rouge_1_scores.append(rouge_1)

            # ROUGE-L (на основе наибольшей общей подпоследовательности)
            def lcs(X, Y):
                m, n = len(X), len(Y)
                L = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    for j in range(n + 1):
                        if i == 0 or j == 0:
                            L[i][j] = 0
                        elif X[i - 1] == Y[j - 1]:
                            L[i][j] = L[i - 1][j - 1] + 1
                        else:
                            L[i][j] = max(L[i - 1][j], L[i][j - 1])
                return L[m][n]

            lcs_length = lcs(pred_words, ref_words)
            rouge_l_precision = lcs_length / len(pred_words) if pred_words else 0.0
            rouge_l_recall = lcs_length / len(ref_words) if ref_words else 0.0
            rouge_l = 2 * rouge_l_precision * rouge_l_recall / (rouge_l_precision + rouge_l_recall) if (
                                                                                                                   rouge_l_precision + rouge_l_recall) > 0 else 0.0
            rouge_l_scores.append(rouge_l)

        return np.mean(rouge_1_scores) if rouge_1_scores else 0.0, np.mean(rouge_l_scores) if rouge_l_scores else 0.0

    def calculate_meteor(self, predictions, references):
        def word_matches(pred, ref):
            matches = sum(1 for word in set(pred.split()) if word in set(ref[0].split()))
            return matches

        meteor_scores = []
        for pred, refs in zip(predictions, references):
            pred_words = pred.split()
            ref_words = refs[0].split()
            matches = word_matches(pred, refs)
            precision = matches / len(pred_words) if pred_words else 0.0
            recall = matches / len(ref_words) if ref_words else 0.0
            meteor = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            meteor_scores.append(meteor)

        return np.mean(meteor_scores) if meteor_scores else 0.0

    def train(self, progress_callback=None, metric_queue=None, run_id=None):
        self.is_training = True
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/run_{run_id}")
        try:
            self.log_architecture_params(run_id)
            self.tokenizer = self.train_tokenizer(run_id)
            self.config.tokenizer_path = f"runs/run_{run_id}/tokenizer.json"
            self.config.pad_token_id = self.tokenizer.token_to_id("[PAD]")
            optimizers = []
            if self.other_params:
                optimizer = torch.optim.Adam(
                    self.other_params,
                    lr=self.config.learning_rate,
                    betas=(self.config.beta1, self.config.beta2),
                    eps=self.config.epsilon
                )
                optimizers.append(optimizer)
            else:
                self.log("Ошибка: Нет параметров модели для оптимизации (other_params пуст)", Fore.RED)
                raise ValueError("No model parameters for optimization")
            if self.dropout_params:
                dropout_optimizer = torch.optim.Adam(
                    self.dropout_params,
                    lr=self.config.dropout_lr,
                    betas=(self.config.beta1, self.config.beta2),
                    eps=self.config.epsilon
                )
                optimizers.append(dropout_optimizer)
                self.log(f"Оптимизатор для dropout_params инициализирован с lr={self.config.dropout_lr}", Fore.BLUE)
            else:
                self.log("Предупреждение: Нет параметров dropout для оптимизации, обучаемый дропаут отключен",
                         Fore.YELLOW)
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("[PAD]"))

            train_dataset, val_dataset = self.data_processor.load_datasets()
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size,
                                                                shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            self.total_steps = len(self.train_dataloader) * self.config.epochs

            start_epoch = self.load_checkpoint(run_id)
            self.model.train()
            global_step = start_epoch * len(self.train_dataloader)
            log_interval = max(1, len(self.train_dataloader) // 10)

            for epoch in range(start_epoch, self.config.epochs):
                self.current_epoch = epoch + 1
                epoch_train_loss = 0.0
                progress_bar = tqdm(enumerate(self.train_dataloader, 1), total=len(self.train_dataloader),
                                    desc=f"Эпоха {self.current_epoch}/{self.config.epochs}")

                for step, (src, tgt) in progress_bar:
                    if not self.is_training:
                        self.log(f"Обучение прервано на эпохе {self.current_epoch}, шаге {step}", Fore.RED)
                        return
                    self.current_step = global_step
                    try:
                        loss = self.process_batch(src, tgt, optimizers, criterion)
                    except Exception as e:
                        self.log(f"Ошибка обработки батча на шаге {step}: {str(e)}", Fore.RED)
                        raise
                    epoch_train_loss += loss
                    self.writer.add_scalar("Loss/train", loss, global_step)
                    self.log_trainable_params(global_step)

                    if step % log_interval == 0:
                        self.log(
                            f"Эпоха {self.current_epoch}/{self.config.epochs}, Шаг {step}/{len(self.train_dataloader)}, Loss: {loss:.4f}",
                            Fore.BLUE)

                    if progress_callback:
                        progress = (global_step + 1) / self.total_steps * 100
                        progress_callback(progress, self.current_epoch, self.config.epochs, step,
                                          len(self.train_dataloader), global_step, self.total_steps)
                    global_step += 1

                epoch_train_loss /= len(self.train_dataloader)
                self.save_checkpoint(self.current_epoch, run_id)

                val_loss, bleu_score, word_count_match, avg_word_len_match, token_repetition, rouge_1, rouge_l, meteor = self.validate(
                    val_dataloader, criterion)
                self.writer.add_scalar("Loss/val", val_loss, self.current_epoch)
                self.writer.add_scalar("BLEU/val", bleu_score, self.current_epoch)
                self.writer.add_scalar("WordCountMatch/val", word_count_match, self.current_epoch)
                self.writer.add_scalar("AvgWordLenMatch/val", avg_word_len_match, self.current_epoch)
                self.writer.add_scalar("TokenRepetition/val", token_repetition, self.current_epoch)
                self.writer.add_scalar("ROUGE-1/val", rouge_1, self.current_epoch)
                self.writer.add_scalar("ROUGE-L/val", rouge_l, self.current_epoch)
                self.writer.add_scalar("METEOR/val", meteor, self.current_epoch)

                if metric_queue:
                    metric_queue.put({
                        'train_loss': epoch_train_loss,
                        'val_loss': val_loss,
                        'bleu': bleu_score,
                        'word_count_match': word_count_match,
                        'avg_word_len_match': avg_word_len_match,
                        'token_repetition': token_repetition,
                        'rouge_1': rouge_1,
                        'rouge_l': rouge_l,
                        'meteor': meteor
                    })
                self.log(
                    f"Эпоха {self.current_epoch} Метрики: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"BLEU: {bleu_score:.4f}, WordCountMatch: {word_count_match:.4f}, AvgWordLenMatch: {avg_word_len_match:.4f}, "
                    f"TokenRepetition: {token_repetition:.4f}, ROUGE-1: {rouge_1:.4f}, ROUGE-L: {rouge_l:.4f}, METEOR: {meteor:.4f}",
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
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        references = []

        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                src_key_padding_mask = self.model.create_padding_mask(src, self.tokenizer.token_to_id("[PAD]"))
                tgt_key_padding_mask = self.model.create_padding_mask(tgt[:, :-1], self.tokenizer.token_to_id("[PAD]"))
                tgt_mask = self.model.create_look_ahead_mask(tgt.size(1) - 1)
                output = self.model(src, tgt[:, :-1], src_key_padding_mask, tgt_mask, tgt_key_padding_mask,
                                    writer=self.writer, global_step=self.current_step)
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
        word_count_match, avg_word_len_match, token_repetition = self.calculate_custom_metrics(predictions, references)
        rouge_1, rouge_l = self.calculate_rouge_metrics(predictions, references)
        meteor = self.calculate_meteor(predictions, references)
        self.model.train()
        return val_loss, bleu_score, word_count_match, avg_word_len_match, token_repetition, rouge_1, rouge_l, meteor

    def stop_training(self):
        self.is_training = False
        self.log("Запрос на остановку обучения", Fore.RED)

    def translate(self, text):
        self.log(f"Начало перевода текста: {text}", Fore.YELLOW)
        try:
            self.model.eval()
            tokens = self.tokenizer.encode(text).ids
            src = torch.tensor([tokens[:self.config.max_len]], dtype=torch.long).to(self.device)
            src = torch.nn.functional.pad(src, (0, self.config.max_len - src.size(1)),
                                          value=self.tokenizer.token_to_id("[PAD]"))
            src_key_padding_mask = self.model.create_padding_mask(src, self.tokenizer.token_to_id("[PAD]"))
            tgt = torch.tensor([[self.tokenizer.token_to_id("[CLS]")]], dtype=torch.long).to(self.device)

            for _ in range(self.config.max_len - 1):
                tgt_key_padding_mask = self.model.create_padding_mask(tgt, self.tokenizer.token_to_id("[PAD]"))
                tgt_mask = self.model.create_look_ahead_mask(tgt.size(1))
                output = self.model(src, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, writer=self.writer,
                                    global_step=self.current_step)
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