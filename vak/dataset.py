from datasets import load_dataset
from configs.training import TrainingConfig
from configs.lora import LoraConfig
from typing import Tuple

class VakDataset:
    def __init__(self, train_cfg: TrainingConfig, seed: LoraConfig):
        self.train_cfg = train_cfg
        self.seed = seed.SEED
        
    def load(self):
        dataset = load_dataset(
            self.train_cfg.DATASET,
            split="train"
        )
        
        dataset = dataset.shuffle(seed=self.seed)
        return dataset
    
    def split(self, dataset) -> Tuple:
        valid_data = dataset.select(range(3000))
        train_data = dataset.select(range(3000, len(dataset)))
        
        return train_data, valid_data
    
    def preprocess(self, dataset, tokenize):
        dataset = dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
        )
        
        return dataset