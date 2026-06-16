"""
Вычисление perplexity для моделей
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
import torch.nn as nn


def compute_perplexity(model, loader, device, vocab_size):
    """Вычисление perplexity на DataLoader"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            output = model(x)
            output = output.permute(0, 2, 1).reshape(-1, vocab_size)
            target = y.reshape(-1)
            loss = criterion(output, target)
            
            num_tokens = y.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


class PerplexityEvaluator:
    """Оценщик perplexity для distilgpt2"""
    
    def __init__(self, model_name="distilgpt2", device="cuda"):
        self.device = device
        
        print("Загрузка distilgpt2...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = GPT2Config.from_pretrained(model_name)
        self.config.loss_type = "cross_entropy"

        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=self.config)
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Модель загружена на {device}")
        print(f"✓ Число параметров: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_perplexity_from_texts(self, texts, max_length=512, batch_size=8):
        """Вычисление perplexity на наборе текстов"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity, avg_loss
    
    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """Генерация текста"""
        self.model.eval()
        
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text