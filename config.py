class Config:
    def __init__(self):
        self.dataset_paths = []
        self.datasets_dir = "datasets/"
        self.model_path = "models/transformer.pt"
        self.tokenizer_path = "models/tokenizer.json"
        self.log_dir = "logs/"
        self.vocab_size = 30000
        self.max_len = 16
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.1
        self.batch_size = 128
        self.epochs = 30
        self.learning_rate = 0.0001
        self.dropout_lr = 0.0001  # New parameter for dropout learning rate
        self.warmup_steps = 4000
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 1e-9
        self.min_chars = 1
        self.max_chars = 51
        self.invalid_chars = r'[^\w\s.,!?-]'
        self.val_split = 0.4
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.tokenizer_type = "BPE"
        self.activation = "gelu"
        self.normalization_type = "layer_norm"
        self.dropout_attn = 0.1
        self.use_learnable_dropout = False
        self.use_learnable_dropout_attn = False
        self.ffn_dim = 512
        self.norm_eps = 1e-6
        self.apply_residual = True
        self.pad_token_id = None

    def validate(self):
        assert self.vocab_size > 0, "vocab_size должен быть положительным"
        assert 0 <= self.dropout < 1, "dropout должен быть в [0, 1)"
        assert 0 <= self.dropout_attn < 1, "dropout_attn должен быть в [0, 1)"
        assert self.d_model > 0, "d_model должен быть положительным"
        assert self.n_heads > 0, "n_heads должен быть положительным"
        assert self.d_model % self.n_heads == 0, "d_model должен быть кратен n_heads"
        assert self.n_layers > 0, "n_layers должен быть положительным"
        assert self.ffn_dim > 0, "ffn_dim должен быть положительным"
        assert self.norm_eps > 0, "norm_eps должен быть положительным"
        assert self.dropout_lr > 0, "dropout_lr должен быть положительным"