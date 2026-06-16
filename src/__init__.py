"""
Модули для сравнения языковых моделей
"""

from src.data_loader import WikiTextLoader
from src.model import LSTMLanguageModel
from src.training import Trainer
from src.evaluation import PerplexityEvaluator

__version__ = "1.0.0"