import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        if self.method == "cbow":
            self._train_cbow(corpus, tokenizer, criterion, optimizer, num_epochs)
        elif self.method == "skipgram":
            self._train_skipgram(corpus, tokenizer, criterion, optimizer, num_epochs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        pass

    def _train_cbow(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
    ) -> None:
        self.train()
        device = next(self.parameters()).device
        for epoch in range(num_epochs):
            total_loss = 0.0
            for sentence in corpus:
                token_ids = tokenizer.encode(sentence, add_special_tokens=False)
                if tokenizer.pad_token_id is not None:
                    token_ids = [token for token in token_ids if token != tokenizer.pad_token_id]
                if len(token_ids) == 0:
                    continue
                
                for i, target in enumerate(token_ids):
                    context = []
                    start = max(0, i - self.window_size)
                    end = min(len(token_ids), i + self.window_size + 1)
                    for j in range(start, end):
                        if j != i:
                            context.append(token_ids[j])
                    if not context:
                        continue
                    context_tensor = torch.tensor(context, dtype=torch.long, device=device)
                    
                    context_embeds = self.embeddings(context_tensor)  # (context_size, d_model)
                    context_mean = context_embeds.mean(dim=0, keepdim=True)  # (1, d_model)
                    logits = self.weight(context_mean)  # (1, vocab_size)
                    target_tensor = torch.tensor([target], dtype=torch.long, device=device)
                    loss = criterion(logits, target_tensor)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            print(f"CBOW Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        pass

    def _train_skipgram(
        self,
        # 구현하세요!
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
    ) -> None:
        # 구현하세요!
        self.train()
        device = next(self.parameters()).device
        for epoch in range(num_epochs):
            total_loss = 0.0
            for sentence in corpus:
                token_ids = tokenizer.encode(sentence, add_special_tokens=False)
                if tokenizer.pad_token_id is not None:
                    token_ids = [token for token in token_ids if token != tokenizer.pad_token_id]
                if len(token_ids) == 0:
                    continue

                for i, center in enumerate(token_ids):
                    start = max(0, i - self.window_size)
                    end = min(len(token_ids), i + self.window_size + 1)
                    for j in range(start, end):
                        if j == i:
                            continue
                        context_word = token_ids[j]
                        center_tensor = torch.tensor([center], dtype=torch.long, device=device)
                        center_embed = self.embeddings(center_tensor)  # (1, d_model)
                        logits = self.weight(center_embed)  # (1, vocab_size)
                        target_tensor = torch.tensor([context_word], dtype=torch.long, device=device)
                        loss = criterion(logits, target_tensor)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
            print(f"Skipgram Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

        pass

    # 구현하세요!
    pass