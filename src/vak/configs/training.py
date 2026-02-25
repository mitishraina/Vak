class TrainingConfig:
    MODEL="HuggingFaceTB/SmolLM-135M"
    DATASET="flytech/python-codes-25k"
    DATA_COLUMN="output"

    SEQ_LEN=512

    MAX_STEPS=2000
    BATCH_SIZE=16
    GR_ACC_STEPS=1
    LR=5e-4
    LR_SCHEDULER_TYPE="cosine"
    WEIGHT_DECAY=0.01
    NUM_WARMUP_STEPS=100
    EVAL_FREQ=100
    SAVE_FREQ=100
    LOG_FREQ=20
    OUTPUT_DIR="./output"
    BF16=True
    FP16=False

    FIM_RATE=0.0
    FIM_SPM_RATE=0.0
    
config = TrainingConfig()