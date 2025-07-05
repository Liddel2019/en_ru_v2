import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import time
import sacrebleu
from colorama import init, Fore, Style

init()

class Trainer:
    def __init__(self, model, data_processor, config):
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        os.makedirs(os.path.dirname(self.config.tokenizer_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.config.model_path) or '.', exist_ok=True)
        self.tokenizer = None  # Tokenizer will be trained in train method
        self.writer = None
        self.is_training = False
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        print(f"===trainer.py===\n{Fore.BLUE}Trainer initialized with device: {self.device}{Style.RESET_ALL}")

    def train_tokenizer(self, run_id):
        print(f"===trainer.py===\n{Fore.YELLOW}Starting tokenizer training for run {run_id}...{Style.RESET_ALL}")
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.config.vocab_size,
                           special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        if not self.config.dataset_paths:
            datasets_info = self.data_processor.check_datasets()
            self.config.dataset_paths = [os.path.join(self.config.datasets_dir, f) for f in datasets_info.keys() if
                                       datasets_info[f]["valid"]]
            print(f"===trainer.py===\n{Fore.CYAN}Dataset paths set: {self.config.dataset_paths}{Style.RESET_ALL}")
        tokenizer.train(self.config.dataset_paths, trainer)
        tokenizer_path = f"runs/run_{run_id}/tokenizer.json"
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"===trainer.py===\n{Fore.GREEN}Tokenizer trained and saved to {tokenizer_path}{Style.RESET_ALL}")
        return tokenizer

    def train(self, progress_callback=None, metric_queue=None, run_id=None):
        self.is_training = True
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/run_{run_id}")
        self.tokenizer = self.train_tokenizer(run_id)  # Train tokenizer for this run
        self.config.tokenizer_path = f"runs/run_{run_id}/tokenizer.json"  # Update tokenizer path
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("[PAD]"))

        train_dataset, val_dataset = self.data_processor.load_datasets()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        self.total_steps = len(train_dataloader) * self.config.epochs
        print(f"===trainer.py===\n{Fore.CYAN}Starting training with {self.config.epochs} epochs, {len(train_dataloader)} steps per epoch, total steps: {self.total_steps}{Style.RESET_ALL}")

        self.model.train()
        global_step = 0
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1
            if not self.is_training:
                print(f"===trainer.py===\n{Fore.RED}Training interrupted at epoch {self.current_epoch}{Style.RESET_ALL}")
                break
            print(f"===trainer.py===\n{Fore.MAGENTA}Starting epoch {self.current_epoch}/{self.config.epochs}{Style.RESET_ALL}")
            epoch_train_loss = 0.0
            for step, (src, tgt) in enumerate(train_dataloader, 1):
                self.current_step = step
                if not self.is_training:
                    print(f"===trainer.py===\n{Fore.RED}Training stopped at step {step} in epoch {self.current_epoch}{Style.RESET_ALL}")
                    break
                src, tgt = src.to(self.device), tgt.to(self.device)
                optimizer.zero_grad()
                output = self.model(src, tgt[:, :-1])
                loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), global_step)
                print(f"===trainer.py===\n{Fore.BLUE}Epoch {self.current_epoch}/{self.config.epochs}, Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Global Step: {global_step}{Style.RESET_ALL}")
                if progress_callback:
                    progress = (global_step + 1) / self.total_steps * 100
                    progress_callback(progress, self.current_epoch, self.config.epochs, step, len(train_dataloader), global_step, self.total_steps)
                global_step += 1

            # Compute Validation Loss
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for src, tgt in val_dataloader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    output = self.model(src, tgt[:, :-1])
                    loss = criterion(output.contiguous().view(-1, self.config.vocab_size), tgt[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
                    val_steps += 1
            val_loss = val_loss / val_steps if val_steps > 0 else 0.0

            # Compute BLEU Score on validation set
            bleu_score = self.compute_bleu(val_dataloader)
            self.model.train()

            # Send metrics to GUI
            if metric_queue:
                metric_queue.put({
                    'train_loss': epoch_train_loss / len(train_dataloader),
                    'val_loss': val_loss,
                    'bleu': bleu_score
                })
            self.writer.add_scalar("Loss/val", val_loss, self.current_epoch)
            self.writer.add_scalar("BLEU/val", bleu_score, self.current_epoch)
            print(f"===trainer.py===\n{Fore.GREEN}Epoch {self.current_epoch} Metrics: Train Loss: {epoch_train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}{Style.RESET_ALL}")

        self.writer.close()
        torch.save(self.model.state_dict(), self.config.model_path)
        print(f"===trainer.py===\n{Fore.GREEN}Training completed, model saved to {self.config.model_path}{Style.RESET_ALL}")

    def compute_bleu(self, dataloader):
        """Compute BLEU score for validation set."""
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                output = self.model(src, tgt[:, :-1])
                pred_tokens = output.argmax(dim=-1).cpu().numpy()
                for pred, ref in zip(pred_tokens, tgt.cpu().numpy()):
                    pred_text = self.tokenizer.decode(pred)
                    ref_text = self.tokenizer.decode(ref[1:])  # Skip [CLS]
                    if ref_text.strip():
                        predictions.append(pred_text)
                        references.append([ref_text])
        bleu = sacrebleu.corpus_bleu(predictions, references).score
        return bleu

    def stop_training(self):
        self.is_training = False
        print(f"===trainer.py===\n{Fore.RED}Training stop requested{Style.RESET_ALL}")

    def translate(self, text):
        print(f"===trainer.py===\n{Fore.YELLOW}Starting translation for text: {text}{Style.RESET_ALL}")
        self.model.eval()
        tokens = self.tokenizer.encode(text).ids
        src = torch.tensor([tokens], dtype=torch.long).to(self.device)
        tgt = torch.tensor([[self.tokenizer.token_to_id("[CLS]")]], dtype=torch.long).to(self.device)

        for i in range(self.config.max_len):
            output = self.model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=-1)
            print(f"===trainer.py===\n{Fore.CYAN}Generated token {i+1}/{self.config.max_len}: {next_token.item()}{Style.RESET_ALL}")
            if next_token.item() == self.tokenizer.token_to_id("[SEP]"):
                print(f"===trainer.py===\n{Fore.GREEN}Translation completed early at token {i+1}{Style.RESET_ALL}")
                break

        translation = self.tokenizer.decode(tgt[0].cpu().numpy())
        print(f"===trainer.py===\n{Fore.GREEN}Translation result: {translation}{Style.RESET_ALL}")
        return translation