from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch

from configs.lora import LoraConfig
from configs.training import TrainingConfig

device_map = "auto"

class VakModel:
    def __init__(self):
        self.train_cfg = TrainingConfig()
        self.lora_cfg = LoraConfig()
        
        set_seed(self.lora_cfg.SEED)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.train_cfg.MODEL,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = None

    def load_model(self):
        if self.lora_cfg.LOAD_IN_4BIT:
            compute_dtype = getattr(torch, self.lora_cfg.BNB_4BIT_COMPUTE_TYPE)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.lora_cfg.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.lora_cfg.USE_NESTED_QUANT
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.train_cfg.MODEL,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
        return self.model
    
    def tokenize(self, example):
        text = example[self.train_cfg.DATA_COLUMN]
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.train_cfg.SEQ_LEN
        )
        
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens