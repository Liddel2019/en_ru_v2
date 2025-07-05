
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
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.1
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.0005
        self.warmup_steps = 4004
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 1e-9
        self.min_chars = 2
        self.max_chars = 200
        self.invalid_chars = r'[^\w\s.,!?-]'
        self.val_split = 0.2  # Added validation split
