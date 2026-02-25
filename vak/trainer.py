from transformers import Trainer, TrainingArguments
from configs.training import TrainingConfig

class VakTrainer:
    def __init__(self):
        self.train_cfg = TrainingConfig()
        
    def build_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.train_cfg.OUTPUT_DIR,
            dataloader_drop_last=True,
            save_strategy="steps",
            max_steps=self.train_cfg.MAX_STEPS,
            eval_steps=self.train_cfg.EVAL_FREQ,
            save_steps=self.train_cfg.SAVE_FREQ,
            num_train_epochs=self.train_cfg.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=self.train_cfg.BATCH_SIZE,
            per_device_eval_batch_size=self.train_cfg.BATCH_SIZE,
            gradient_accumulation_steps=self.train_cfg.GR_ACC_STEPS,
            gradient_checkpointing=True,
            learning_rate=self.train_cfg.LR,
            lr_scheduler_kwargs=self.train_cfg.LR_SCHEDULER_TYPE,
            warmup_steps=self.train_cfg.NUM_WARMUP_STEPS,
            logging_steps=self.train_cfg.LOG_FREQ,
            weight_decay=self.train_cfg.WEIGHT_DECAY,
            fp16=self.train_cfg.FP16,
            bf16=self.train_cfg.BF16,
            push_to_hub=True,
        )
        
        return training_args
    
    def build_trainer(self, model, tokenizer, train_data, valid_data):
        training_args = self.build_training_args()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            tokenizer=tokenizer
        )
        
        return trainer