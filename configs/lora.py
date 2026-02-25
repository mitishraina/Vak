class LoraConfig:
    LORA_R=8
    LORA_ALPHA=32
    LORA_DROPOUT=0.0
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    USE_NESTED_QUANT=True
    BNB_4BIT_COMPUTE_TYPE="bfloat16"
    LOAD_IN_4BIT=True
    BNB_4BIT_QUANT_TYPE="nf4"
    SEED=0
    
config = LoraConfig()