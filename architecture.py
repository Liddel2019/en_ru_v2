import torch
import torch.nn as nn
import math
from colorama import init, Fore, Style

init()

class LearnableDropout(nn.Module):
    def __init__(self, dropout_rate=0.1, dropout_name="dropout"):
        super(LearnableDropout, self).__init__()
        self.dropout_rate = nn.Parameter(torch.tensor(dropout_rate), requires_grad=True)
        self.dropout_name = dropout_name

    def forward(self, x, writer=None, global_step=None):
        if not self.training:
            return x
        rate = torch.clamp(self.dropout_rate, 0.0, 0.999)
        mask = torch.rand_like(x) > rate
        output = x * mask / (1.0 - rate)
        if writer is not None and global_step is not None and (global_step % 10 == 0 or global_step == 0):
            writer.add_scalar(f"Dropout/{self.dropout_name}", rate.item(), global_step)
            if self.dropout_rate.grad is not None:
                writer.add_scalar(f"Gradients/Dropout/{self.dropout_name}", self.dropout_rate.grad.item(), global_step)
            else:
                writer.add_text(f"Dropout/{self.dropout_name}/warning", f"No gradient for {self.dropout_name} at step {global_step}", global_step)
        return output

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model)
        )
        self.dropout = LearnableDropout(config.dropout, dropout_name="encoder_dropout") if config.use_learnable_dropout else nn.Dropout(config.dropout)
        self.attn_dropout = LearnableDropout(config.dropout_attn, dropout_name="encoder_attn_dropout") if config.use_learnable_dropout_attn else nn.Dropout(config.dropout_attn)
        self.apply_residual = config.apply_residual

    def forward(self, src, src_key_padding_mask=None, writer=None, global_step=None):
        attn_output, _ = self.attention(src, src, src, key_padding_mask=src_key_padding_mask)
        attn_output = self.attn_dropout(attn_output, writer=writer, global_step=global_step) if isinstance(self.attn_dropout, LearnableDropout) else self.attn_dropout(attn_output)
        src = self.norm1(src + attn_output if self.apply_residual else attn_output)
        ffn_output = self.ffn(src)
        ffn_output = self.dropout(ffn_output, writer=writer, global_step=global_step) if isinstance(self.dropout, LearnableDropout) else self.dropout(ffn_output)
        src = self.norm2(src + ffn_output if self.apply_residual else ffn_output)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=0.0, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model)
        )
        self.dropout = LearnableDropout(config.dropout, dropout_name="decoder_dropout") if config.use_learnable_dropout else nn.Dropout(config.dropout)
        self.self_attn_dropout = LearnableDropout(config.dropout_attn, dropout_name="decoder_self_attn_dropout") if config.use_learnable_dropout_attn else nn.Dropout(config.dropout_attn)
        self.cross_attn_dropout = LearnableDropout(config.dropout_attn, dropout_name="decoder_cross_attn_dropout") if config.use_learnable_dropout_attn else nn.Dropout(config.dropout_attn)
        self.apply_residual = config.apply_residual

    def forward(self, tgt, src_output, tgt_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None, writer=None, global_step=None):
        attn_output, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        attn_output = self.self_attn_dropout(attn_output, writer=writer, global_step=global_step) if isinstance(self.self_attn_dropout, LearnableDropout) else self.self_attn_dropout(attn_output)
        tgt = self.norm1(tgt + attn_output if self.apply_residual else attn_output)
        cross_output, _ = self.cross_attention(tgt, src_output, src_output, key_padding_mask=src_key_padding_mask)
        cross_output = self.cross_attn_dropout(cross_output, writer=writer, global_step=global_step) if isinstance(self.cross_attn_dropout, LearnableDropout) else self.cross_attn_dropout(cross_output)
        tgt = self.norm2(tgt + cross_output if self.apply_residual else cross_output)
        ffn_output = self.ffn(tgt)
        ffn_output = self.dropout(ffn_output, writer=writer, global_step=global_step) if isinstance(self.dropout, LearnableDropout) else self.dropout(ffn_output)
        tgt = self.norm3(tgt + ffn_output if self.apply_residual else ffn_output)
        return tgt

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.n_layers)])
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps) if config.normalization_type == "layer_norm" else nn.Identity()
        self.to(self.device)
        print(f"===architecture.py===\n{Fore.BLUE}TransformerModel инициализирован с устройством={self.device}{Style.RESET_ALL}")

    def create_padding_mask(self, seq, pad_token):
        return (seq == pad_token).to(self.device)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(self.device)
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None, writer=None, global_step=None):
        src_key_padding_mask = self.create_padding_mask(src, self.config.pad_token_id) if src_key_padding_mask is None else src_key_padding_mask
        tgt_key_padding_mask = self.create_padding_mask(tgt, self.config.pad_token_id) if tgt_key_padding_mask is None else tgt_key_padding_mask
        if tgt_mask is None:
            tgt_mask = self.create_look_ahead_mask(tgt.size(1))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        src = self.embedding(src) * math.sqrt(self.config.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.config.d_model)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        for layer in self.encoder_layers:
            src = layer(src, src_key_padding_mask, writer=writer if self.config.use_learnable_dropout or self.config.use_learnable_dropout_attn else None, global_step=global_step)
        src_output = src

        for layer in self.decoder_layers:
            tgt = layer(tgt, src_output, tgt_mask, tgt_key_padding_mask, src_key_padding_mask, writer=writer if self.config.use_learnable_dropout or self.config.use_learnable_dropout_attn else None, global_step=global_step)
        output = self.norm(tgt)
        output = self.fc(output)
        return output