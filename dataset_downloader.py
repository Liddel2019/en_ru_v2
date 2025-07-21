import requests
import os
import tqdm
import logging
import validators
from colorama import init, Fore, Style
from config import Config

init()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_download.log'), logging.StreamHandler()]
)

def validate_url(url):
    return validators.url(url) is True

def download_dataset(dataset_name, url, progress_callback=None):
    config = Config()
    save_path = os.path.join(config.datasets_dir, dataset_name)

    if os.path.exists(save_path):
        logging.info(f"Датасет {dataset_name} уже существует по пути {save_path}")
        print(f"===dataset_downloader.py===\n{Fore.YELLOW}Датасет {dataset_name} уже существует по пути {save_path}{Style.RESET_ALL}")
        return True, "Файл уже существует"

    if not validate_url(url):
        logging.warning(f"Недействительный URL для {dataset_name}: {url}")
        print(f"===dataset_downloader.py===\n{Fore.RED}Недействительный URL для {dataset_name}: {url}{Style.RESET_ALL}")
        return False, "Недействительный URL"

    logging.info(f"Попытка загрузки {dataset_name} с {url}")
    print(f"===dataset_downloader.py===\n{Fore.BLUE}Попытка загрузки {dataset_name} с {url}{Style.RESET_ALL}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(save_path, 'wb') as file, tqdm.tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=f"Загрузка {dataset_name}") as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                if progress_callback:
                    progress_callback(bar.n, block_size, total_size)

        logging.info(f"Успешно загружен {dataset_name} с {url}")
        print(f"===dataset_downloader.py===\n{Fore.GREEN}Загрузка завершена для {dataset_name} с {url}{Style.RESET_ALL}")
        return True, "Загрузка успешно завершена"
    except requests.exceptions.RequestException as e:
        logging.error(f"Не удалось загрузить {dataset_name} с {url}: {str(e)}")
        print(f"===dataset_downloader.py===\n{Fore.RED}Не удалось загрузить с {url}: {str(e)}{Style.RESET_ALL}")
        return False, f"Ошибка загрузки: {str(e)}"

DATASETS = {
    "opus_en_ru.txt.zip": {
        "urls": [
            "https://object.pouta.csc.fi/OPUS-MT/en-ru/data/en-ru.txt.zip",
            "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.en-ru.txt.zip",
            "http://data.statmt.org/opus/en-ru.txt.zip"
        ],
        "description": "Параллельный корпус OPUS английский-русский"
    },
    "tatoeba_en_ru.tsv.bz2": {
        "urls": [
            "https://downloads.tatoeba.org/exports/per_language/rus/rus-eng_sentences.tsv.bz2",
            "https://archive.tatoeba.org/rus-eng_sentences.tsv.bz2",
            "http://mirror.tatoeba.org/exports/rus-eng_sentences.tsv.bz2"
        ],
        "description": "Пары предложений Tatoeba английский-русский"
    },
    "unpc_en_ru.zip": {
        "urls": [
            "https://conferences.unite.un.org/UN_Corpus/UNv1.0.en-ru.zip",
            "http://data.statmt.org/opus/UNPC/en-ru/UNv1.0.en-ru.zip",
            "https://archive.org/download/UN_Corpus/UNv1.0.en-ru.zip"
        ],
        "description": "Параллельный корпус ООН английский-русский"
    },
    "wmt19_en_ru.txt.gz": {
        "urls": [
            "http://data.statmt.org/wmt19/translation-task/news-commentary-v14.en-ru.gz",
            "https://www.statmt.org/wmt19/translation-task/news-commentary-v14.en-ru.gz",
            "http://archive.statmt.org/wmt19/news-commentary-v14.en-ru.gz"
        ],
        "description": "Датасет перевода WMT 2019 английский-русский"
    },
    "news_commentary_en_ru.txt.gz": {
        "urls": [
            "http://data.statmt.org/news-commentary/v15/en-ru.txt.gz",
            "https://www.statmt.org/wmt20/translation-task/news-commentary-v15.en-ru.txt.gz",
            "http://archive.statmt.org/news-commentary/v15/en-ru.txt.gz"
        ],
        "description": "Параллельный корпус новостей английский-русский"
    }
}

def download_all_datasets(progress_callback=None):
    results = {}
    for dataset_name, dataset_info in DATASETS.items():
        logging.info(f"Обработка датасета: {dataset_name} - {dataset_info['description']}")
        print(f"===dataset_downloader.py===\n{Fore.CYAN}Обработка датасета: {dataset_name} - {dataset_info['description']}{Style.RESET_ALL}")
        for url in dataset_info["urls"]:
            success, message = download_dataset(dataset_name, url, progress_callback)
            if success:
                results[dataset_name] = {"success": True, "message": message}
                break
        else:
            results[dataset_name] = {"success": False, "message": "Все URL не работают"}
    return results