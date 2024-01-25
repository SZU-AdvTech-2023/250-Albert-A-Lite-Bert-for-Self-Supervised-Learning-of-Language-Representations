from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import Trainer, TrainingArguments


dataset = load_dataset("./imdb")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
max_length = 512  # 根据需要设置最大长度

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=max_length)

train_data = dataset["train"].map(tokenize_batch, batched=True)
test_data = dataset["test"].map(tokenize_batch, batched=True)

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)



training_args = TrainingArguments(
    output_dir="./albert_imdb",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()
model.save_pretrained("./albert_imdb_model_cpu")
