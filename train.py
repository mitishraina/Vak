from vak.model import VakModel
from vak.dataset import VakDataset
from vak.peft import VakPeft
from vak.trainer import VakTrainer

vak_model = VakModel()
model = vak_model.load_model()

dataset_builder = VakDataset(vak_model.train_cfg, vak_model.lora_cfg.SEED)
dataset = dataset_builder.load()
train_data, valid_data = dataset_builder.split(dataset)

train_data = dataset_builder.preprocess(train_data, vak_model.tokenize_function)
valid_data = dataset_builder.preprocess(valid_data, vak_model.tokenize_function)

vak_peft = VakPeft  ()
model = vak_peft.prepare_model(model)
model = vak_peft.apply_lora(model)

vak_trainer = VakTrainer()
trainer = vak_trainer.build_trainer(
    model,
    vak_model.tokenizer,
    train_data,
    valid_data
)

trainer.train()

model.save_pretrained("outputs/vak-adapter")