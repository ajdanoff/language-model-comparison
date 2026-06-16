"""
Вспомогательные функции
"""

import torch
import yaml


def load_config(config_path="config.yaml"):
    """Загрузка конфигурации"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    """Определение устройства"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU не доступен, используется CPU")
    
    return device


def save_model(model, path):
    """Сохранение модели"""
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в {path}")


def load_model(model, path, device):
    """Загрузка модели"""
    model.load_state_dict(torch.load(path, map=device))
    model = model.to(device)
    print(f"Модель загружена из {path}")
    return model