import requests
import os
import tqdm
from config import Config
from colorama import init, Fore, Style

init()


def download_dataset(dataset_name, url, progress_callback=None):
    config = Config()
    save_path = os.path.join(config.datasets_dir, dataset_name)
    if os.path.exists(save_path):
        print(f"{Fore.YELLOW}Dataset {dataset_name} already exists at {save_path}{Style.RESET_ALL}")
        return True, "File already exists"

    print(f"{Fore.BLUE}Starting download of {dataset_name} from {url}{Style.RESET_ALL}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with open(save_path, 'wb') as file, tqdm.tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {dataset_name}"
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                if progress_callback:
                    progress_callback(bar.n, block_size, total_size)
        print(f"{Fore.GREEN}Download completed for {dataset_name}{Style.RESET_ALL}")
        return True, "Download completed successfully"
    except Exception as e:
        print(f"{Fore.RED}Download failed for {dataset_name}: {str(e)}{Style.RESET_ALL}")
        return False, f"Download error: {str(e)}"