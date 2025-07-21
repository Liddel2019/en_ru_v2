import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import re
from tokenizers import Tokenizer
from colorama import init, Fore, Style
from sklearn.model_selection import train_test_split

init()


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(
            f"===data.py===\n{Fore.BLUE}TranslationDataset инициализирован с {len(data)} примерами, max_len={max_len}{Style.RESET_ALL}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        # Токенизация исходного и целевого текста
        src_tokens = self.tokenizer.encode(src_text).ids
        tgt_tokens = self.tokenizer.encode(tgt_text).ids

        # Обрезка или паддинг до max_len
        src_tokens = src_tokens[:self.max_len]
        src_tokens += [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(src_tokens))

        # Целевые токены включают [CLS] и [SEP], поэтому max_len-2 для токенов текста
        tgt_tokens = tgt_tokens[:self.max_len - 2]
        tgt_tokens = [self.tokenizer.token_to_id("[CLS]")] + tgt_tokens + [self.tokenizer.token_to_id("[SEP]")]
        tgt_tokens += [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(tgt_tokens))

        # Проверка длины последовательностей
        if len(src_tokens) != self.max_len or len(tgt_tokens) != self.max_len:
            raise ValueError(
                f"Ошибка: длина src_tokens={len(src_tokens)} или tgt_tokens={len(tgt_tokens)} не равна max_len={self.max_len}")

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.length_filters = {"Short": (1, 50), "Medium": (51, 100), "Long": (101, 200)}
        print(
            f"===data.py===\n{Fore.BLUE}DataProcessor инициализирован с datasets_dir={self.config.datasets_dir}{Style.RESET_ALL}")

    def check_datasets(self):
        datasets_info = {}
        os.makedirs(self.config.datasets_dir, exist_ok=True)

        for file in os.listdir(self.config.datasets_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.config.datasets_dir, file)
                print(f"===data.py===\n{Fore.CYAN}Проверка датасета: {file}{Style.RESET_ALL}")
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
                    if not {"en", "ru"}.issubset(df.columns):
                        datasets_info[file] = {"valid": False, "num_rows": 0,
                                               "reason": "Отсутствуют колонки 'en' или 'ru'"}
                        print(f"===data.py===\n{Fore.RED}{file}: Отсутствуют колонки{Style.RESET_ALL}")
                        continue
                    if df.empty:
                        datasets_info[file] = {"valid": False, "num_rows": 0, "reason": "Датасет пуст"}
                        print(f"===data.py===\n{Fore.RED}{file}: Пустой датасет{Style.RESET_ALL}")
                        continue
                    invalid_pattern = re.compile(self.config.invalid_chars)
                    valid_rows = 0
                    for _, row in df.iterrows():
                        en_text = str(row["en"]).strip()
                        ru_text = str(row["ru"]).strip()
                        if not en_text or not ru_text:  # Проверка на пустые строки
                            continue
                        if (len(en_text) >= self.config.min_chars and len(ru_text) >= self.config.min_chars and
                                len(en_text) <= self.config.max_chars and len(ru_text) <= self.config.max_chars and
                                not invalid_pattern.search(en_text) and not invalid_pattern.search(ru_text)):
                            valid_rows += 1
                    datasets_info[file] = {"valid": valid_rows > 0, "num_rows": valid_rows, "reason": ""}
                    print(f"===data.py===\n{Fore.GREEN}{file}: {valid_rows} валидных строк{Style.RESET_ALL}")
                except Exception as e:
                    datasets_info[file] = {"valid": False, "num_rows": 0, "reason": f"Ошибка чтения: {str(e)}"}
                    print(f"===data.py===\n{Fore.RED}{file}: Ошибка - {str(e)}{Style.RESET_ALL}")

        available_urls = {
            "Common_Crawl.csv": "https://example.com/datasets/Common_Crawl.csv",
            "OPUS_Tatoeba.csv": "https://example.com/datasets/OPUS_Tatoeba.csv",
            "WMT19_en-ru.csv": "https://example.com/datasets/WMT19_en-ru.csv"
        }
        for name, url in available_urls.items():
            if name not in datasets_info:
                datasets_info[name] = {"valid": False, "num_rows": 0, "reason": "Доступен для загрузки", "url": url}
                print(f"===data.py===\n{Fore.YELLOW}{name}: Доступен для загрузки по {url}{Style.RESET_ALL}")
        return datasets_info

    def load_datasets(self):
        data = []
        for path in self.config.dataset_paths:
            if os.path.exists(path):
                print(f"===data.py===\n{Fore.CYAN}Загрузка датасета из {path}{Style.RESET_ALL}")
                try:
                    df = pd.read_csv(path, encoding='utf-8', sep=',')
                    for _, row in df.iterrows():
                        en_text = str(row["en"]).strip()
                        ru_text = str(row["ru"]).strip()
                        if not en_text or not ru_text:  # Пропуск пустых строк
                            continue
                        text_len = max(len(en_text), len(ru_text))
                        valid_length = any(
                            min_len <= text_len <= max_len for min_len, max_len in self.length_filters.values())
                        if (valid_length and
                                len(en_text) >= self.config.min_chars and len(ru_text) >= self.config.min_chars and
                                len(en_text) <= self.config.max_chars and len(ru_text) <= self.config.max_chars):
                            data.append((en_text, ru_text))
                except Exception as e:
                    print(f"===data.py===\n{Fore.RED}Ошибка загрузки {path}: {str(e)}{Style.RESET_ALL}")
            else:
                print(f"===data.py===\n{Fore.RED}Путь {path} не существует, пропускаем{Style.RESET_ALL}")

        if not data:
            raise ValueError("Нет валидных данных для загрузки")

        train_data, val_data = train_test_split(data, test_size=self.config.val_split, random_state=42)
        tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        print(
            f"===data.py===\n{Fore.GREEN}Тренировочный датасет загружен с {len(train_data)} примерами, Валидационный датасет с {len(val_data)} примерами{Style.RESET_ALL}")
        return (TranslationDataset(train_data, tokenizer, self.config.max_len),
                TranslationDataset(val_data, tokenizer, self.config.max_len))

    def download_dataset(self, dataset_name, url, progress_callback=None):
        from dataset_downloader import download_dataset
        return download_dataset(dataset_name, url, progress_callback)