from datasets import load_dataset
from src.vak.configs.training import TrainingConfig

cfg = TrainingConfig()

dataset = load_dataset(cfg.DATASET)
