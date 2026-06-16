"""
Основной скрипт для запуска проекта
"""

import torch
import yaml
from src.data_loader import get_data_loaders
from src.model import LSTMLanguageModel, count_parameters
from src.training import Trainer
from src.evaluation import PerplexityEvaluator
from src.utils import get_device, load_config


def main():
    """Главная функция"""
    print("="*60)
    print("Сравнение языковых моделей: LSTM vs GRU vs distilgpt2")
    print("="*60)
    
    # Загрузка конфигурации
    config = load_config("project/config.yaml")
    
    # Определение устройства
    device = get_device()
    
    # Загрузка данных
    print("\n=== 1. Загрузка данных ===")
    train_loader, val_loader, vocab = get_data_loaders("project/config.yaml")
    
    config['model']['vocab_size'] = len(vocab)
    
    # === 2. Обучение LSTM ===
    print("\n=== 2. Обучение LSTM ===")
    lstm_model = LSTMLanguageModel(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        rnn_type="LSTM"
    )
    
    lstm_trainer = Trainer(lstm_model, train_loader, val_loader, device, config)
    lstm_results = lstm_trainer.train(num_epochs=config['training']['num_epochs'])
    lstm_results['num_params'] = count_parameters(lstm_model)
    
    # === 3. Обучение GRU ===
    print("\n=== 3. Обучение GRU ===")
    gru_model = LSTMLanguageModel(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        rnn_type="GRU"
    )
    
    gru_trainer = Trainer(gru_model, train_loader, val_loader, device, config)
    gru_results = gru_trainer.train(num_epochs=config['training']['num_epochs'])
    gru_results['num_params'] = count_parameters(gru_model)
    
    # === 4. Оценка distilgpt2 ===
    print("\n=== 4. Оценка distilgpt2 ===")
    evaluator = PerplexityEvaluator(model_name="distilgpt2", device=device)
    
    # Загрузка тестовых текстов
    from src.data_loader import WikiTextLoader
    loader = WikiTextLoader("config.yaml")
    test_texts = loader.load_dataset(split="test")
    
    distil_ppl, distil_loss = evaluator.compute_perplexity_from_texts(test_texts)
    
    # === 5. Финальное сравнение ===
    print("\n=== 5. Финальное сравнение ===")
    print(f"{'Модель':<15} | {'Параметры':>15} | {'Perplexity':>18} | {'Время/эпоху':>15}")
    print("-"*65)
    print(f"LSTM            | {lstm_results['num_params']:>15,} | {lstm_results['val_ppls'][-1]:>18.4f} | {lstm_results['avg_time_per_epoch']:>14.2f}с")
    print(f"GRU             | {gru_results['num_params']:>15,} | {gru_results['val_ppls'][-1]:>18.4f} | {gru_results['avg_time_per_epoch']:>14.2f}с")
    print(f"distilgpt2      | {sum(p.numel() for p in evaluator.model.parameters()):>15,} | {distil_ppl:>18.4f} | {'N/A':>15}")
    
    print("\n✓ Проект завершён успешно!")


if __name__ == "__main__":
    main()