"""
Обучение и валидация моделей
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


def compute_loss(output, target, vocab_size):
    """
    Вычисление loss
    
    output: (batch_size, seq_length, vocab_size)
    target: (batch_size, seq_length)
    """
    output = output.permute(0, 2, 1).reshape(-1, vocab_size)
    target = target.reshape(-1)
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)


def train_epoch(model, loader, optimizer, device, vocab_size, max_norm=1.0):
    """Обучение одной эпохи"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    model.train()
    total_loss = 0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = compute_loss(output, y, vocab_size)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    epoch_duration = end_time - start_time
    
    return total_loss / num_batches, epoch_duration


def validate(model, loader, device, vocab_size):
    """Валидация"""
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = compute_loss(output, y, vocab_size)
            
            total_loss += loss.item()
            num_batches += 1
    
    val_loss = total_loss / num_batches
    val_ppl = np.exp(val_loss)
    
    return val_loss, val_ppl


class Trainer:
    """Класс для обучения модели"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
        self.train_losses = []
        self.val_losses = []
        self.val_ppls = []
        self.epoch_times = []
    
    def train(self, num_epochs=5):
        """Обучение модели"""
        print(f"Начало обучения на {num_epochs} эпох ...")
        print(f"Используется: {self.device}")
        
        for epoch in range(num_epochs):
            train_loss, epoch_duration = train_epoch(
                self.model, 
                self.train_loader, 
                self.optimizer, 
                self.device,
                self.config['model']['vocab_size'],
                max_norm=self.config['training']['max_norm']
            )
            
            val_loss, val_ppl = validate(
                self.model, 
                self.val_loader, 
                self.device,
                self.config['model']['vocab_size']
            )
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ppls.append(val_ppl)
            self.epoch_times.append(epoch_duration)
            
            print(f"Эпоха {epoch + 1} / {num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"PPL: {val_ppl:.4f} | Time: {epoch_duration:.2f}с")
        
        avg_time = np.mean(self.epoch_times)
        print(f"\Среднее время на эпоху: {avg_time:.2f}с")
        print(f"\Обучение завершено!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ppls': self.val_ppls,
            'avg_time_per_epoch': avg_time
        }