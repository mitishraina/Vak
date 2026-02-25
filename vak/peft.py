from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer
from configs.lora import LoraConfig

class VakPeft:
    def __init__(self):
        self.lora_cfg = LoraConfig()
        
    def prepare_model(self, model):
        if self.lora_cfg.LOAD_IN_4BIT:
            model = prepare_model_for_kbit_training(model)
            
        return model
    
    def apply_lora(self, model):
        peft_config = LoraConfig(
            lora_alpha=self.lora_cfg.LORA_ALPHA,
            lora_dropout=self.lora_cfg.LORA_DROPOUT,
            r=self.lora_cfg.LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_cfg.LORA_TARGET_MODULES
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model