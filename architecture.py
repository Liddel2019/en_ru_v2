import torch
import torch.nn as nn
import math
from colorama import init, Fore, Style

init()

class LearnableDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(LearnableDropout, self).__init__()
        self.dropout_rate = nn.Parameter(torch.tensor(dropout_rate), requires_grad=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        rate = torch.clamp(self.dropout_rate, 0.0, 0.999)
        return self.dropout(x) if self.training else x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class CustomAttention(nn.Module):
    def __init__(self, config):
        super(CustomAttention, self).__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.attention_type = config.attention_type
        self.dropout = LearnableDropout(config.dropout_attn) if config.use_learnable_dropout_attn else nn.Dropout(config.dropout_attn)
        self.d_k = config.d_model // config.n_heads

        if self.attention_type == "scaled_dot_product":
            self.scale = math.sqrt(self.d_k)
        elif self.attention_type == "additive":
            self.W_q = nn.Linear(self.d_model, self.d_model)
            self.W_k = nn.Linear(self.d_model, self.d_model)
            self.W_v = nn.Linear(self.d_model, self.d_model)
            self.W_o = nn.Linear(self.d_model, self.d_model)
            self.W_a = nn.Linear(self.d_model, 1)  # Добавлен слой для проекции в скалярное внимание
        elif self.attention_type == "multi_head":
            self.multihead_attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=config.dropout_attn, batch_first=True)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        if self.attention_type == "scaled_dot_product":
            query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, value)
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        elif self.attention_type == "additive":
            query = self.W_q(query)  # [batch_size, seq_len, d_model]
            key = self.W_k(key)      # [batch_size, seq_len, d_model]
            value = self.W_v(value)  # [batch_size, seq_len, d_model]
            scores = torch.tanh(query + key)  # [batch_size, seq_len, d_model]
            scores = self.W_a(scores).squeeze(-1)  # [batch_size, seq_len]
            if mask is not None:
                scores = scores.unsqueeze(1).masked_fill(mask == 0, -1e9)  # [batch_size, 1, seq_len]
            else:
                scores = scores.unsqueeze(1)  # [batch_size, 1, seq_len]
            attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, value)  # [batch_size, 1, d_model]
            output = self.W_o(output.squeeze(1)).unsqueeze(1)  # [batch_size, seq_len, d_model]
            output = output.expand(batch_size, seq_len, self.d_model)  # Восстанавливаем seq_len
        else:  # multi_head
            output, _ = self.multihead_attn(query, key, value, attn_mask=mask)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = CustomAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else \
                     nn.BatchNorm1d(config.d_model, eps=config.norm_eps) if config.normalization_type == "batch_norm" else nn.Identity()
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else \
                     nn.BatchNorm1d(config.d_model, eps=config.norm_eps) if config.normalization_type == "batch_norm" else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model)
        )
        self.dropout = LearnableDropout(config.dropout) if config.use_learnable_dropout else nn.Dropout(config.dropout)
        self.apply_residual = config.apply_residual

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output if self.apply_residual else attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output if self.apply_residual else ffn_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_len)

        if config.layer_type == "transformer":
            self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        elif config.layer_type == "feed_forward":
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_model, config.ffn_dim),
                    nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                    nn.Linear(config.ffn_dim, config.d_model),
                    nn.Dropout(config.dropout)
                ) for _ in range(config.n_layers)
            ])
        elif config.layer_type == "convolutional":
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(config.d_model, config.ffn_dim, kernel_size=3, padding=1),
                    nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                    nn.Conv1d(config.ffn_dim, config.d_model, kernel_size=3, padding=1),
                    nn.Dropout(config.dropout)
                ) for _ in range(config.n_layers)
            ])

        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else \
                    nn.BatchNorm1d(config.d_model, eps=config.norm_eps) if config.normalization_type == "batch_norm" else nn.Identity()
        self.to(self.device)
        print(f"===architecture.py===\n{Fore.BLUE}TransformerModel инициализирован с устройством={self.device}{Style.RESET_ALL}")

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.config.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.config.d_model)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        if self.config.layer_type == "transformer":
            output = src
            for layer in self.layers:
                output = layer(output, src_key_padding_mask)
            src_output = output
            output = tgt
            for layer in self.layers:
                output = layer(output, tgt_mask)
            tgt_output = output
        else:
            output = src
            for layer in self.layers:
                if self.config.layer_type == "convolutional":
                    output = output.transpose(1, 2)
                    output = layer(output)
                    output = output.transpose(1, 2)
                else:
                    output = layer(output)
            src_output = output
            output = tgt
            for layer in self.layers:
                if self.config.layer_type == "convolutional":
                    output = output.transpose(1, 2)
                    output = layer(output)
                    output = output.transpose(1, 2)
                else:
                    output = layer(output)
            tgt_output = output

        output = self.norm(tgt_output)
        output = self.fc(output)
        return output