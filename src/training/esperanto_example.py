#https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb

import torch

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments



# load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

# initialize model
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)


# make a dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)



# make a trainer
training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)



trainer.train()


trainer.save_model("./EsperBERTo")
