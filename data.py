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
        print(f"===data.py===\n{Fore.BLUE}TranslationDataset initialized with {len(data)} samples, max_len={max_len}{Style.RESET_ALL}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = self.tokenizer.encode(src_text).ids
        tgt_tokens = self.tokenizer.encode(tgt_text).ids
        src_tokens = src_tokens[:self.max_len] + [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_len] + [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(tgt_tokens))
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.length_filters = {"Short": (1, 50), "Medium": (51, 100), "Long": (101, 200)}  # Default filters
        print(f"===data.py===\n{Fore.BLUE}DataProcessor initialized with datasets_dir={config.datasets_dir}{Style.RESET_ALL}")

    def check_datasets(self):
        datasets_info = {}
        if not os.path.exists(self.config.datasets_dir):
            os.makedirs(self.config.datasets_dir, exist_ok=True)
            print(f"===data.py===\n{Fore.YELLOW}Created datasets directory: {self.config.datasets_dir}{Style.RESET_ALL}")

        for file in os.listdir(self.config.datasets_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.config.datasets_dir, file)
                print(f"===data.py===\n{Fore.CYAN}Checking dataset: {file}{Style.RESET_ALL}")
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
                    if not {"en", "ru"}.issubset(df.columns):
                        datasets_info[file] = {"valid": False, "num_rows": 0, "reason": "Missing 'en' or 'ru' columns"}
                        print(f"===data.py===\n{Fore.RED}{file}: Missing columns{Style.RESET_ALL}")
                        continue
                    if df.empty:
                        datasets_info[file] = {"valid": False, "num_rows": 0, "reason": "Dataset is empty"}
                        print(f"===data.py===\n{Fore.RED}{file}: Empty dataset{Style.RESET_ALL}")
                        continue
                    invalid_pattern = re.compile(self.config.invalid_chars)
                    valid_rows = 0
                    for _, row in df.iterrows():
                        en_text = str(row["en"]).strip()
                        ru_text = str(row["ru"]).strip()
                        if (len(en_text) >= self.config.min_chars and len(ru_text) >= self.config.min_chars and
                            len(en_text) <= self.config.max_chars and len(ru_text) <= self.config.max_chars and
                            not invalid_pattern.search(en_text) and not invalid_pattern.search(ru_text)):
                            valid_rows += 1
                    datasets_info[file] = {"valid": valid_rows > 0, "num_rows": valid_rows,
                                         "reason": "Invalid data" if valid_rows == 0 else ""}
                    print(f"===data.py===\n{Fore.GREEN}{file}: {valid_rows} valid rows{Style.RESET_ALL}")
                except Exception as e:
                    datasets_info[file] = {"valid": False, "num_rows": 0, "reason": f"Read error: {str(e)}"}
                    print(f"===data.py===\n{Fore.RED}{file}: Error - {str(e)}{Style.RESET_ALL}")

        available_urls = {
            "Common_Crawl.csv": "https://example.com/datasets/Common_Crawl.csv",
            "OPUS_Tatoeba.csv": "https://example.com/datasets/OPUS_Tatoeba.csv",
            "WMT19_en-ru.csv": "https://example.com/datasets/WMT19_en-ru.csv"
        }
        for name, url in available_urls.items():
            if name not in datasets_info:
                datasets_info[name] = {"valid": False, "num_rows": 0, "reason": "Available for download", "url": url}
                print(f"===data.py===\n{Fore.YELLOW}{name}: Available for download at {url}{Style.RESET_ALL}")

        return datasets_info

    def load_datasets(self):
        """Load datasets and split into training and validation sets, applying length filters."""
        data = []
        for path in self.config.dataset_paths:
            if os.path.exists(path):
                print(f"===data.py===\n{Fore.CYAN}Loading dataset from {path}{Style.RESET_ALL}")
                df = pd.read_csv(path, encoding='utf-8', sep=',')
                for _, row in df.iterrows():
                    en_text = str(row["en"]).strip()
                    ru_text = str(row["ru"]).strip()
                    # Check if text length falls within any selected filter range
                    text_len = max(len(en_text), len(ru_text))
                    valid_length = False
                    for min_len, max_len in self.length_filters.values():
                        if min_len <= text_len <= max_len:
                            valid_length = True
                            break
                    if (valid_length and
                        len(en_text) >= self.config.min_chars and len(ru_text) >= self.config.min_chars and
                        len(en_text) <= self.config.max_chars and len(ru_text) <= self.config.max_chars):
                        data.append((en_text, ru_text))
            else:
                print(f"===data.py===\n{Fore.RED}Path {path} does not exist, skipping{Style.RESET_ALL}")

        # Split data into training and validation sets
        train_data, val_data = train_test_split(data, test_size=self.config.val_split, random_state=42)
        tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        print(f"===data.py===\n{Fore.GREEN}Training dataset loaded with {len(train_data)} samples, Validation dataset with {len(val_data)} samples{Style.RESET_ALL}")
        return (TranslationDataset(train_data, tokenizer, self.config.max_len),
                TranslationDataset(val_data, tokenizer, self.config.max_len))

    def download_dataset(self, dataset_name, url, progress_callback=None):
        """Wrapper for dataset_downloader.download_dataset to maintain compatibility."""
        from dataset_downloader import download_dataset
        return download_dataset(dataset_name, url, progress_callback)