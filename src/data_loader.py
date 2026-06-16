"""
Загрузка и preprocessing данных WikiText-2
"""

import re
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import yaml
import numpy as np
import matplotlib.pyplot as plt


class WikiTextLoader:
    """Загрузка и токенизация датасета WikiText-2"""
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация loader'а"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_dataset = self.config['dataset']
        self.clean_string = self._clean_string
        self.vocab = None
    
    def _clean_string(self, text):
        """Очистка текста"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_dataset(self, split="train", max_texts_count=7000):
        """Загрузка датасета WikiText-2"""
        print("=== Загрузка датасета WikiText-2 ===")
        
        dataset = load_dataset(
            self.config_dataset['name'],
            self.config_dataset['config'],
            split=split
        )
        
        texts = [line for line in dataset["text"] 
                 if len(line.split()) >= self.config['model']['seq_length']]
        cleaned_texts = list(map(self._clean_string, texts))
        
        print(f"✓ Загружено текстов: {len(cleaned_texts)}")
        return cleaned_texts
    
    def tokenize_text(self, text):
        """Токенизация текста"""
        return text.split()
    
    def build_vocab(self, texts, min_freq=2):
        """Построение словаря"""
        print("\n=== Построение словаря ===")
        
        word_counter = Counter()
        for text in texts:
            tokens = self.tokenize_text(text)
            word_counter.update(tokens)
        
        vocab = {"<unk>": 0, "<pad>": 1}
        idx = 2
        for word, count in word_counter.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        
        self.vocab = vocab
        print(f"Размер словаря: {len(vocab)}")
        return vocab
    
    def text_to_indices(self, text, vocab):
        """Конвертация текста в индексы"""
        tokens = self.tokenize_text(text)
        indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        return indices
    
    def create_sequences(self, indices, seq_length):
        """Создание последовательностей"""
        sequences = []
        targets = []
        for i in range(0, len(indices) - seq_length):
            seq = indices[i:i + seq_length]
            target = indices[i + 1:i + seq_length + 1]
            sequences.append(seq)
            targets.append(target)
        return sequences, targets


class WikiTextDataset(Dataset):
    """PyTorch Dataset для WikiText"""
    
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


class DataLoaderHelper:
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.loader = WikiTextLoader(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def concatenate_all_texts(self, texts_list, vocab):
        all_indices = []
        for text in texts_list:
            indices = self.loader.text_to_indices(text, vocab)
            all_indices.extend(indices)
        return all_indices
    
    def load_train_texts(self):
        # Загрузка данных
        train_texts = self.loader.load_dataset(
            split=self.config['dataset']['split'],
            max_texts_count=self.config['model']['max_texts_count']
        )
        return train_texts
    
    @classmethod
    def train_texts_analysis(cls, train_texts):
        # 2a. Примеры текстов
        print("\n--- Примеры текстов (первые 5) ---")
        for i in range(5):
            print(f"\nТекст #{i+1} (первые 300 символов):")
            print(train_texts[i][:300] + "...")
        
        # 2b. Подсчёт статистики
        # # Число предложений (по точкам)
        sentence_counts = [len(re.split(r'\.', text)) - 1 for text in train_texts]
        sentence_counts = [c if c > 0 else 1 for c in sentence_counts]  # заменяем 0 на 1
        
        print("\n=== Статистика по количеству предложений в тексте ===")
        print(f"Среднее: {np.mean(sentence_counts):.2f}")
        print(f"Медиана: {np.median(sentence_counts):.2f}")
        print(f"Минимум: {min(sentence_counts)}")
        print(f"Максимум: {max(sentence_counts)}")
        
        # Количество слов в текстах
        word_counts = [len(text.split()) for text in train_texts]
        
        print("\n=== Статистика по количеству слов в тексте ===")
        print(f"Среднее: {np.mean(word_counts):.2f}")
        print(f"Медиана: {np.median(word_counts):.2f}")
        print(f"5-й перцентиль: {np.percentile(word_counts, 5):.2f}")
        print(f"95-й перцентиль: {np.percentile(word_counts, 95):.2f}")
        print(f"Минимум: {min(word_counts)}")
        print(f"Максимум: {max(word_counts)}")
        print(f"Стандартное отклонение: {np.std(word_counts):.2f}")
        
        # 2c. Гистограмма распределения длины
        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=50, edgecolor='black')
        plt.title("Распределение количества слов в текстах")
        plt.xlabel("Количество слов")
        plt.ylabel("Частота")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def split_train_val(self, train_texts):
        train_texts, val_texts = train_test_split(
            train_texts, 
            test_size=self.config['training']['val_size'], 
            random_state=self.config['training']['random_seed']
        )
        return train_texts, val_texts
    
    def build_vocab(self, train_texts):
        vocab = self.loader.build_vocab(train_texts, min_freq=self.config['model']['min_freq'])
        return vocab
    
    def train_val_datasets(self, train_texts, val_texts, vocab):
        train_all_indices = self.concatenate_all_texts(train_texts, vocab)
        val_all_indices = self.concatenate_all_texts(val_texts, vocab)
        
        # Создание последовательностей
        train_sequences, train_targets = self.loader.create_sequences(
            train_all_indices, self.config['model']['seq_length']
        )
        
        val_sequences, val_targets = self.loader.create_sequences(
            val_all_indices, self.config['model']['seq_length']
        )
        
        # Создание Dataset и DataLoader
        train_dataset = WikiTextDataset(train_sequences, train_targets)
        val_dataset = WikiTextDataset(val_sequences, val_targets)
        return train_dataset, val_dataset
    
    def get_data_loaders(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=False
        )
        return train_loader, val_loader
    
    def get_data_loaders_pipeline(self, analysis=True):
        train_texts = self.load_train_texts()
        if analysis:
            self.train_texts_analysis(train_texts)
        train_texts, val_texts = self.split_train_val(train_texts)
        vocab = self.build_vocab(train_texts)
        train_dataset, val_dataset = self.train_val_datasets(train_texts, val_texts, vocab)
        train_loader, val_loader = self.get_data_loaders(train_dataset, val_dataset)
        return train_loader, val_loader, vocab
    
        
# запускать без хелпера (анализа данных)
def get_data_loaders(config_path="config.yaml"):
    """Функция для получения DataLoader'ов"""
    loader = WikiTextLoader(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Загрузка данных
    train_texts = loader.load_dataset(
        split=config['dataset']['split'],
        max_texts_count=config['model']['max_texts_count']
    )
    
    # Разделение на train/val
    train_texts, val_texts = train_test_split(
        train_texts, 
        test_size=config['training']['val_size'], 
        random_state=config['training']['random_seed']
    )
    
    # Построение словаря
    vocab = loader.build_vocab(train_texts, min_freq=config['model']['min_freq'])
    
    # Конвертация в индексы
    def concatenate_all_texts(texts_list, vocab):
        all_indices = []
        for text in texts_list:
            indices = loader.text_to_indices(text, vocab)
            all_indices.extend(indices)
        return all_indices
    
    train_all_indices = concatenate_all_texts(train_texts, vocab)
    val_all_indices = concatenate_all_texts(val_texts, vocab)
    
    # Создание последовательностей
    train_sequences, train_targets = loader.create_sequences(
        train_all_indices, config['model']['seq_length']
    )
    val_sequences, val_targets = loader.create_sequences(
        val_all_indices, config['model']['seq_length']
    )
    
    # Создание Dataset и DataLoader
    train_dataset = WikiTextDataset(train_sequences, train_targets)
    val_dataset = WikiTextDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False
    )
    
    return train_loader, val_loader, vocab