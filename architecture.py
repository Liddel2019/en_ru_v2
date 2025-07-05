import torch
import torch.nn as nn
from colorama import init, Fore, Style

init()

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dropout=config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"===architecture.py===\n{Fore.BLUE}TransformerModel initialized with vocab_size={config.vocab_size}, d_model={config.d_model}, device={self.device}{Style.RESET_ALL}")

    def forward(self, src, tgt):
        print(f"===architecture.py===\n{Fore.CYAN}Forward pass with src shape: {src.shape}, tgt shape: {tgt.shape}{Style.RESET_ALL}")
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        output = self.transformer(src_embed, tgt_embed)
        output = self.fc(output)
        print(f"===architecture.py===\n{Fore.CYAN}Output shape: {output.shape}{Style.RESET_ALL}")
        return output