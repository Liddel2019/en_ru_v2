import tkinter as tk
from gui import GUI
from config import Config
from trainer import Trainer
from data import DataProcessor
from architecture import TransformerModel
from colorama import init, Fore, Style

init()

def main():
    print(f"===main.py===\n{Fore.BLUE}Starting application...{Style.RESET_ALL}")
    config = Config()
    data_processor = DataProcessor(config)
    model = TransformerModel(config)
    trainer = Trainer(model, data_processor, config)
    root = tk.Tk()
    app = GUI(root, trainer, config, data_processor)
    print(f"===main.py===\n{Fore.GREEN}GUI application started{Style.RESET_ALL}")
    root.mainloop()

if __name__ == "__main__":
    main()